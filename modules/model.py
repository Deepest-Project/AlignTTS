import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths


class Prenet(nn.Module):
    def __init__(self, hp):
        super(Prenet, self).__init__()
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text):
        B, L = text.size(0), text.size(1)
        x = self.Embedding(text).transpose(0,1)
        x += self.pe[:L].unsqueeze(1)
        x = self.dropout(x).transpose(0,1)
        return x

    
class FFT(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, n_layers):
        super(FFT, self).__init__()
        self.FFT_layers = nn.ModuleList([TransformerEncoderLayer(d_model=hidden_dim,
                                                                 nhead=n_heads,
                                                                 dim_feedforward=ff_dim)
                                      for _ in range(n_layers)])

    def forward(self, x, lengths):
        alignments = []
        x = x.transpose(0,1)
        mask = get_mask_from_lengths(lengths)
        for layer in self.FFT_layers:
            x, align = layer(x, src_key_padding_mask=mask)
            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)

        return x.transpose(0,1), alignments

    
class DurationPredictor(nn.Module):
    def __init__(self, hp):
        super(DurationPredictor, self).__init__()
        self.FFT = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.linear = Linear(hp.hidden_dim, 1)

    def forward(self, encoder_input, text_lengths):
        x = encoder_input.transpose(0,1)
        x = self.FFT(encoder_input, text_lengths).transpose(0,1)
        x = self.linear(x)
        return x

    
class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        self.Prenet = Prenet(hp)
        self.FFT_lower = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.FFT_upper = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.MDN = nn.Sequential(Linear(hp.hidden_dim, 256),
                                 nn.LayerNorm(256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 Linear(256, 2*hp.n_mel_channels))
        self.DurationPredictor = DurationPredictor(hp)
        self.Projection = Linear(hp.hidden_dim, hp.n_mel_channels)

    def get_mu_sigma(self, hidden_states):
        mu_sigma = self.MDN(hidden_states)
        return mu_sigma
    
    def get_duration(self, encoder_input, text_lengths):
        durations = self.DurationPredictor(encoder_input, text_lengths).exp()
        return durations
    
    def get_melspec(self, hidden_states, probable_path, mel_lengths, inference=False):
        # probable_path: B, T
        indices = hidden_states.new_tensor(torch.arange(probable_path.max()+1).view(1,1,-1)) # 1, 1, L
        path_onehot = 1.0*(hidden_states.new_tensor(indices==probable_path.unsqueeze(-1))) # B, T, L
        hidden_states_expanded = torch.matmul(path_onehot, hidden_states)
        hidden_states_expanded += self.Prenet.pe[:hidden_states_expanded.size(1)].unsqueeze(1).transpose(0,1)
        mel_out = self.Projection(self.FFT_upper(hidden_states_expanded, mel_lengths)[0]).transpose(1,2)
        return mel_out
    
    def forward(self, text, melspec, probable_path, text_lengths, mel_lengths, criterion, stage):
        text = text[:,:text_lengths.max().item()]
        melspec = melspec[:,:,:mel_lengths.max().item()]
        if stage==0:
            encoder_input = self.Prenet(text)
            hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
            mu_sigma = self.get_mu_sigma(hidden_states)
            mdn_loss, _, _, _ = criterion(mu_sigma, melspec, text_lengths, mel_lengths)
            return mdn_loss
        
        elif stage==1:
            encoder_input = self.Prenet(text)
            hidden_states, _ = FFT_lower(encoder_input, text_lengths)
            mel_out = get_melspec(hidden_states, probable_path, mel_lengths)
            fft_loss = nn.L1Loss()(mel_out, melspec)
            return fft_loss
        
        elif stage==2:
            encoder_input = self.Prenet(text)
            hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
            mu_sigma = self.get_mu_sigma(hidden_states)
            mdn_loss, log_prob_matrix, _, _ = criterion(mu_sigma, melspec, text_lengths, mel_lengths)
            
            probable_path = self.viterbi(log_prob_matrix, text_lengths, mel_lengths) # B, T
            mel_out = self.get_melspec(hidden_states, probable_path, mel_lengths)
            fft_loss = nn.L1Loss()(mel_out, melspec)
            return mdn_loss + fft_loss
        
        elif stage==3:
            encoder_input = self.Prenet(text)
            durations_out = self.get_duration(encoder_input.data, text_lengths) # gradient cut
            
            path_onehot = 1.0*(hidden_states.new_tensor(torch.arange(probable_path.max()+1)).unsqueeze(-1)==probable_path) # B, T, L
            target_durations = self.align2duration(path_onehot, mel_mask)
            duration_loss = nn.L2Loss(torch.log(durations_out), torch.log(target_durations))
            return duration_loss
        
        
    def inference(self, text, alpha=1.0):
        text_lengths = torch.tensor([1, text.size(1)])
        encoder_input = self.Prenet(text)
        hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
        durations = self.get_duration(encoder_input, text_lengths)
        mel_out = self.get_melspec(hidden_states, durations, mel_lengths, inference=True)
        return mel_out, durations
    
    
    def viterbi(self, log_prob_matrix, text_lengths, mel_lengths):
        B, L, T = log_prob_matrix.size()
        log_beta = log_prob_matrix.new_ones(B, L, T)*(-1e15)
        log_beta[:, 0, 0] = log_prob_matrix[:, 0, 0]

        for t in range(1, T):
            prev_step = torch.cat([log_beta[:, :, t-1:t], F.pad(log_beta[:, :, t-1:t], (0,0,1,-1), value=-1e15)], dim=-1).max(dim=-1)[0]
            log_beta[:, :, t] = prev_step+log_prob_matrix[:, :, t]

        curr_rows = text_lengths-1
        curr_cols = mel_lengths-1
        path = [curr_rows*1.0]
        for _ in range(T-1):
            is_go = log_beta[torch.arange(B), (curr_rows-1).to(torch.long), (curr_cols-1).to(torch.long)]\
                     > log_beta[torch.arange(B), (curr_rows).to(torch.long), (curr_cols-1).to(torch.long)]
            curr_rows = F.relu(curr_rows-1.0*is_go+1.0)-1.0
            curr_cols = F.relu(curr_cols-1+1.0)-1.0
            path.append(curr_rows*1.0)

        path.reverse()
        path = torch.stack(path, -1)
        return path
        
    
    def align2duration(self, alignments, mel_lengths):
        ids = alignments.new_tensor( torch.arange(alignments.size(2)) )
        max_ids = torch.max(alignments, dim=2)[1].unsqueeze(-1)
        mel_mask = get_mask_from_lengths(mel_lengths)
        one_hot = 1.0*(ids==max_ids)
        one_hot.masked_fill_(mel_mask.unsqueeze(2), 0)
        
        durations = torch.sum(one_hot, dim=1)

        return durations
