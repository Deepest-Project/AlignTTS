import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths

from datetime import datetime
from time import sleep

class Prenet(nn.Module):
    def __init__(self, hp):
        super(Prenet, self).__init__()
        # B, L -> B, L, D
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
        # B, L, D -> B, L, D
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
        self.Prenet = Prenet(hp)
        self.FFT = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, 2)
        self.linear = Linear(hp.hidden_dim, 1)

    def forward(self, text, text_lengths):
        # B, L -> B, L
        encoder_input = self.Prenet(text)
        x = self.FFT(encoder_input, text_lengths)[0]
        x = self.linear(x).squeeze(-1)
        return x

    
class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.Prenet = Prenet(hp)
        self.FFT_lower = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.FFT_upper = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.MDN = nn.Sequential(Linear(hp.hidden_dim, hp.hidden_dim),
                                 nn.LayerNorm(hp.hidden_dim),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 Linear(hp.hidden_dim, 2*hp.n_mel_channels))
        self.DurationPredictor = DurationPredictor(hp)
        self.Projection = Linear(hp.hidden_dim, hp.n_mel_channels)

    def get_mu_sigma(self, hidden_states):
        mu_sigma = self.MDN(hidden_states)
        return mu_sigma
    
    def get_duration(self, text, text_lengths):
        durations = self.DurationPredictor(text, text_lengths).exp()
        return durations
    
    def get_melspec(self, hidden_states, align, mel_lengths):
        hidden_states_expanded = torch.matmul(align.transpose(1,2), hidden_states)
        hidden_states_expanded += self.Prenet.pe[:hidden_states_expanded.size(1)].unsqueeze(1).transpose(0,1)
        mel_out = torch.sigmoid(self.Projection(self.FFT_upper(hidden_states_expanded, mel_lengths)[0]).transpose(1,2))
        return mel_out
    
    def forward(self, text, melspec, align, text_lengths, mel_lengths, criterion, stage, log_viterbi=False, cpu_viterbi=False):
        text = text[:,:text_lengths.max().item()]
        melspec = melspec[:,:,:mel_lengths.max().item()]
        
        if stage==0:
            encoder_input = self.Prenet(text)
            hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
            mu_sigma = self.get_mu_sigma(hidden_states)
            mdn_loss, _ = criterion(mu_sigma, melspec, text_lengths, mel_lengths)
            return mdn_loss
        
        elif stage==1:
            align = align[:, :text_lengths.max().item(), :mel_lengths.max().item()]
            encoder_input = self.Prenet(text)
            hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
            mel_out = self.get_melspec(hidden_states, align, mel_lengths)
            
            mel_mask = ~get_mask_from_lengths(mel_lengths)
            melspec = melspec.masked_select(mel_mask.unsqueeze(1))
            mel_out = mel_out.masked_select(mel_mask.unsqueeze(1))
            fft_loss = nn.L1Loss()(mel_out, melspec)
            
            return fft_loss
        
        elif stage==2:
            encoder_input = self.Prenet(text)
            hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
            mu_sigma = self.get_mu_sigma(hidden_states)
            mdn_loss, log_prob_matrix = criterion(mu_sigma, melspec, text_lengths, mel_lengths)
            
            before = datetime.now()
            if cpu_viterbi:
                align = self.viterbi_cpu(log_prob_matrix, text_lengths.cpu(), mel_lengths.cpu()) # B, T
            else:
                align = self.viterbi(log_prob_matrix, text_lengths, mel_lengths) # B, T
            after = datetime.now()    
            
            if log_viterbi:
                time_delta = after - before
                print(f'Viterbi took {time_delta.total_seconds()} secs')

            mel_out = self.get_melspec(hidden_states, align, mel_lengths)
            
            mel_mask = ~get_mask_from_lengths(mel_lengths)
            melspec = melspec.masked_select(mel_mask.unsqueeze(1))
            mel_out = mel_out.masked_select(mel_mask.unsqueeze(1))
            fft_loss = nn.L1Loss()(mel_out, melspec)
            
            return mdn_loss + fft_loss
        
        elif stage==3:
            align = align[:, :text_lengths.max().item(), :mel_lengths.max().item()]
            duration_out = self.get_duration(text, text_lengths) # gradient cut
            duration_target = align.sum(-1)
            
            duration_mask = ~get_mask_from_lengths(text_lengths)
            duration_target = duration_target.masked_select(duration_mask)
            duration_out = duration_out.masked_select(duration_mask)
            duration_loss = nn.MSELoss()(torch.log(duration_out), torch.log(duration_target))

            return duration_loss
        
        
    def inference(self, text, alpha=1.0):
        text_lengths = text.new_tensor([text.size(1)])
        encoder_input = self.Prenet(text)
        hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
        durations = self.get_duration(text, text_lengths)
        durations = torch.round(durations*alpha).to(torch.long)
        durations[durations<=0]=1
        T=int(durations.sum().item())
        mel_lengths = text.new_tensor([T])
        hidden_states_expanded = torch.repeat_interleave(hidden_states, durations[0], dim=1)
        hidden_states_expanded += self.Prenet.pe[:hidden_states_expanded.size(1)].unsqueeze(1).transpose(0,1)
        mel_out = torch.sigmoid(self.Projection(self.FFT_upper(hidden_states_expanded, mel_lengths)[0]).transpose(1,2))
        
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
        
        indices = path.new_tensor(torch.arange(path.max()+1).view(1,1,-1)) # 1, 1, L
        align = 1.0*(path.new_tensor(indices==path.unsqueeze(-1))) # B, T, L
        
        for i in range(align.size(0)):
            pad= T-mel_lengths[i]
            align[i] = F.pad(align[i], (0,0,-pad,pad))
            
        return align.transpose(1,2)
    
    def fast_viterbi(self, log_prob_matrix, text_lengths, mel_lengths):
        B, L, T = log_prob_matrix.size()
        
        _log_prob_matrix = log_prob_matrix.cpu()

        curr_rows = text_lengths.cpu().to(torch.long)-1
        curr_cols = mel_lengths.cpu().to(torch.long)-1
        
        path = [curr_rows*1]       
        
        for _ in range(T-1):
#             print(curr_rows-1)
#             print(curr_cols-1)
            is_go = _log_prob_matrix[torch.arange(B), curr_rows-1, curr_cols-1]\
                     > _log_prob_matrix[torch.arange(B), curr_rows, curr_cols-1]
#             curr_rows = F.relu(curr_rows-1*is_go+1)-1
#             curr_cols = F.relu(curr_cols)-1
            curr_rows = F.relu(curr_rows-1*is_go+1)-1
            curr_cols = F.relu(curr_cols-1+1)-1
            path.append(curr_rows*1)

        path.reverse()
        path = torch.stack(path, -1)
        
        indices = path.new_tensor(torch.arange(path.max()+1).view(1,1,-1)) # 1, 1, L
        align = 1.0*(path.new_tensor(indices==path.unsqueeze(-1))) # B, T, L
        
        for i in range(align.size(0)):
            pad= T-mel_lengths[i]
            align[i] = F.pad(align[i], (0,0,-pad,pad))
            
        return align.transpose(1,2)
    
    def viterbi_cpu(self, log_prob_matrix, text_lengths, mel_lengths):
        
        original_device = log_prob_matrix.device

        B, L, T = log_prob_matrix.size()
        
        _log_prob_matrix = log_prob_matrix.cpu()
        
        log_beta = _log_prob_matrix.new_ones(B, L, T)*(-1e15)
        log_beta[:, 0, 0] = _log_prob_matrix[:, 0, 0]

        for t in range(1, T):
            prev_step = torch.cat([log_beta[:, :, t-1:t], F.pad(log_beta[:, :, t-1:t], (0,0,1,-1), value=-1e15)], dim=-1).max(dim=-1)[0]
            log_beta[:, :, t] = prev_step+_log_prob_matrix[:, :, t]

        curr_rows = text_lengths-1
        curr_cols = mel_lengths-1
        path = [curr_rows*1]
        for _ in range(T-1):
            is_go = log_beta[torch.arange(B), curr_rows-1, curr_cols-1]\
                     > log_beta[torch.arange(B), curr_rows, curr_cols-1]
            curr_rows = F.relu(curr_rows - 1 * is_go + 1) - 1
            curr_cols = F.relu(curr_cols) - 1
            path.append(curr_rows*1)

        path.reverse()
        path = torch.stack(path, -1)
        
        indices = path.new_tensor(torch.arange(path.max()+1).view(1,1,-1)) # 1, 1, L
        align = 1.0*(path.new_tensor(indices==path.unsqueeze(-1))) # B, T, L
        
        for i in range(align.size(0)):
            pad= T-mel_lengths[i]
            align[i] = F.pad(align[i], (0,0,-pad,pad))
            
        return align.transpose(1,2).to(original_device)
    
