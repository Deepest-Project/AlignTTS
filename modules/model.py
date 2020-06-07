import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths


class FFT(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, n_layers):
        super(FFT, self).__init__()
        self.FFT_layers = nn.ModuleList([TransformerEncoderLayer(d_model=hidden_dim,
                                                                 nhead=n_heads,
                                                                 dim_feedforward=ff_dim)
                                      for _ in range(n_layers)])

    def forward(self, x, lengths):
        alignments = []
        mask = get_mask_from_lengths(lengths)
        for layer in self.FFT_layers:
            x, align = layer(x, src_key_padding_mask=mask)
            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)

        return x, alignments


class MixDensityNetwork(nn.Module):
    def __init__(self, hp):
        super(MixDensityNetwork, self).__init__()
        self.linear = nn.ModuleList([Linear(hp.hidden_dim, 256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     Linear(256, 2*hp.n_mel_channels),
                                     nn.LayerNorm(2*hp.n_mel_channels),
                                     nn.ReLU()])

    def forward(self, hidden_states):
        mu_sigma = self.linear(hidden_states)
        return mu_sigma


class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, n_layers):
        super(DurationPredictor, self).__init__()
        self.FFT = FFT(hidden_dim, n_heads, ff_dim, n_layers)
        self.linear = Linear(hidden_dim,hidden_dim)

    def forward(self, encoder_input, text_lengths):
        durations = self.linear(self.FFT(encoder_input, text_lengths))
        return durations
    

class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.dropout = nn.Dropout(0.1)
        
        self.FFT_lower = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.FFT_upper = FFT(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.MDN = MixDensityNetwork(hp)
        self.DurationPredictor = DurationPredictor(hp.hidden_dim, hp.n_heads, hp.ff_dim, hp.n_layers)
        self.Projection = Linear(hp.hidden_dim, hp.n_mel_channels)
        
        
    def get_hidden_states(self, text, text_lengths):
        ### Size ###
        B, L, T = text.size(0), text.size(1), melspec.size(2)
        
        ### Prepare Encoder Input ###
        encoder_input = self.Embedding(text).transpose(0,1)
        encoder_input += self.pe[:L].unsqueeze(1)
        encoder_input = self.dropout(encoder_input)
        hidden_states, enc_alignments = FFT_lower(encoder_input, text_lengths)
        
        return hidden_states, enc_alignments
    
    def get_mu_sigma(self, hidden_states):
        mu_sigma = self.MDN(hidden_states)
        return mu_sigma
    
    def get_duration(self, encoder_input, text_lengths):
        durations = self.DurationPredictor(encoder_input, text_lengths)
        return durations
    
    def get_melspec(self, hidden_states, text_lengths)
    
    def forward(self, text, melspec, gate, text_lengths, mel_lengths, criterion, step):
        hidden_states = get_hidden_states(self, text, text_lengths)
        
        if step==0:
            mu_sigma=get_mu_sigma(self, hidden_states)
            mdn_loss = MDN_LOSS(mu_sigma, melspec)
            return mdn_loss
        
        elif step==1:
            ## return FFT_LOSS
        elif step==2:
            ## return MDN_LOSS + FFT_LOSS
        elif step==3:
            ## return duration_LOSS
        
    def inference(self, text, alpha=1.0):
        ### Prepare Inference ###
        text_lengths = torch.tensor([1, text.size(1)])
        
        ### Prepare Inputs ###
        encoder_input = self.Embedding(text).transpose(0,1)
        encoder_input += self.pe[:text.size(1)].unsqueeze(1)
        
        ### Speech Synthesis ###
        text_mask = text.new_zeros(1,text.size(1)).to(torch.bool)
        hidden_states = self.FFT_lower(encoder_input, text_lengths)
            
        ### Duration Predictor ###
        durations = self.DurationPredictor(hidden_states.transpose(0,1))
        hidden_states_expanded = self.LR(hidden_states, durations, alpha, inference=True)
        hidden_states_expanded += self.pe[:hidden_states_expanded.size(0)].unsqueeze(1)
        mel_lengths = hidden_states_expanded.size(0)
        
        hidden_states_expanded = self.FFT_upper(hidden_states_expanded, mel_lengths)
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1,2)
        
        return mel_out, durations

    
    def align2duration(self, alignments, mel_mask):
        ids = alignments.new_tensor( torch.arange(alignments.size(2)) )
        max_ids = torch.max(alignments, dim=2)[1].unsqueeze(-1)
        
        one_hot = 1.0*(ids==max_ids)
        one_hot.masked_fill_(mel_mask.unsqueeze(2), 0)
        
        durations = torch.sum(one_hot, dim=1)

        return durations
    
    
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