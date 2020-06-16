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
        durations = self.DurationPredictor(encoder_input, text_lengths)
        return durations
    
    def get_melspec(self, hiddens_states, durations, mel_lengths, inference=False):
        hidden_states = hidden_states.transpose(0,1)
        hidden_states_expanded = self.LR(hidden_states, durations, alpha, inference=inference)
        hidden_states_expanded += self.Prenet.pe[:hidden_states_expanded.size(0)].unsqueeze(1)
        mel_out = self.Projection(self.FFT_upper(hidden_states_expanded, mel_lengths).transpose(0,1)).transpose(1,2)
        return mel_out
    
    def forward(self, text, melspec, durations, text_lengths, mel_lengths, criterion, stage):
        text = text[:,:text_lengths.max().item()]
        melspec = melspec[:,:,:mel_lengths.max().item()]
        if stage==0:
            encoder_input = self.Prenet(text)
            hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
            mu_sigma = self.get_mu_sigma(hidden_states)
            mdn_loss, _, _, _ = criterion(mu_sigma, melspec, text_lengths, mel_lengths)
            return mdn_loss
        
        elif step==1:
            encoder_input = self.Prenet(text)
            hidden_states, _ = FFT_lower(encoder_input, text_lengths)
            mel_out = get_melspec(hidden_states, durations, mel_lengths)
            fft_loss = nn.L1Loss()(mel_out, melspec)
            return fft_loss
        
        elif step==2:
            encoder_input = self.Prenet(text)
            hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
            mu_sigma = self.get_mu_sigma(hidden_states)
            mdn_loss = criterion(mu_sigma, melspec, mel_lengths)
            
            mel_out = self.get_melspec(hidden_states, durations, mel_lengths)
            fft_loss = nn.L1Loss()(mel_out, melspec)
            return mdn_loss + fft_loss
        
        elif step==3:
            encoder_input = self.Prenet(text)
            durations_out = self.get_duration(encoder_input, text_lengths)
            duration_loss = nn.L2Loss(torch.log(durations_out), torch.log(durations))
            return duration_loss
        
    def inference(self, text, alpha=1.0):
        text_lengths = torch.tensor([1, text.size(1)])
        encoder_input = self.Prenet(text)
        hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
        durations = self.get_duration(encoder_input, text_lengths)
        mel_out = self.get_melspec(hidden_states, durations, mel_lengths, inference=True)
        return mel_out, durations
    
    
    def LR(self, hidden_states, durations, alpha=1.0, inference=False):
        L, B, D = hidden_states.size()
        durations = torch.round(durations*alpha).to(torch.long)
        if inference:
            durations[durations<=0]=1
        T=int(torch.sum(durations, dim=-1).max().item())
        expanded = hidden_states.new_zeros(T, B, D)
        
        for i, d in enumerate(durations):
            mel_len = torch.sum(d).item()
            expanded[:mel_len, i] = torch.repeat_interleave(hidden_states[:, i],
                                                            d,
                                                            dim=0)
            
        return expanded