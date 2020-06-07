import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths


class CBAD(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size,
                 stride,
                 padding,
                 bias,
                 activation,
                 dropout):
        super(CBAD, self).__init__()
        self.conv = Conv1d(in_dim,
                           out_dim,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=bias,
                           w_init_gain=activation)

        self.bn = nn.BatchNorm1d(out_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        out = self.dropout(x)

        return out


class Prenet_D(nn.Module):
    def __init__(self, hp):
        super(Prenet_D, self).__init__()
        self.linear1 = Linear(hp.n_mel_channels,
                              hp.dprenet_dim,
                              w_init_gain='relu')
        self.linear2 = Linear(hp.dprenet_dim, hp.dprenet_dim, w_init_gain='relu')
        self.linear3 = Linear(hp.dprenet_dim, hp.hidden_dim)

    def forward(self, x):
        # Set training==True following tacotron2
        x = F.dropout(F.relu(self.linear1(x)), p=0.5, training=True)
        x = F.dropout(F.relu(self.linear2(x)), p=0.5, training=True)
        x = self.linear3(x)
        return x


class PostNet(nn.Module):
    def __init__(self, hp):
        super(PostNet, self).__init__()
        conv_list = [CBAD(in_dim=hp.n_mel_channels,
                          out_dim=hp.postnet_dim,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False,
                          activation='tanh',
                          dropout=0.5)]
        
        for _ in range(hp.n_postnet_layers-2):
            conv_list.append(CBAD(in_dim=hp.postnet_dim,
                                  out_dim=hp.postnet_dim,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2,
                                  bias=False,
                                  activation='tanh',
                                  dropout=0.5))
            
        conv_list.append(nn.Sequential(nn.Conv1d(hp.postnet_dim,
                                                 hp.n_mel_channels,
                                                 kernel_size=5,
                                                 padding=2,
                                                 bias=False),
                                       nn.BatchNorm1d(hp.n_mel_channels),
                                       nn.Dropout(0.5)))
        
        self.conv=nn.ModuleList(conv_list)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        self.Prenet_D = Prenet_D(hp)
        
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.dropout = nn.Dropout(0.1)
        
        self.Encoder = nn.ModuleList([TransformerEncoderLayer(d_model=hp.hidden_dim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])

        self.Decoder = nn.ModuleList([TransformerDecoderLayer(d_model=hp.hidden_dim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])

        self.Projection = Linear(hp.hidden_dim, hp.n_mel_channels)
        self.Postnet = PostNet(hp)
        self.Stop = nn.Linear(hp.n_mel_channels, 1)
        
        
    def outputs(self, text, melspec, text_lengths, mel_lengths):
        ### Size ###
        B, L, T = text.size(0), text.size(1), melspec.size(2)
        
        ### Prepare Encoder Input ###
        encoder_input = self.Embedding(text).transpose(0,1)
        encoder_input += self.alpha1*(self.pe[:L].unsqueeze(1))
        encoder_input = self.dropout(encoder_input)

        ### Prepare Decoder Input ###
        mel_input = F.pad(melspec, (1,-1)).transpose(1,2)
        decoder_input = self.Prenet_D(mel_input).transpose(0,1)
        decoder_input += self.alpha2*(self.pe[:T].unsqueeze(1))
        decoder_input = self.dropout(decoder_input)

        ### Prepare Masks ###
        text_mask = get_mask_from_lengths(text_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        diag_mask = torch.triu(melspec.new_ones(T,T)).transpose(0, 1)
        diag_mask[diag_mask == 0] = -float('inf')
        diag_mask[diag_mask == 1] = 0

        ### Transformer Encoder ###
        memory = encoder_input
        enc_alignments = []
        for layer in self.Encoder:
            memory, enc_align = layer(memory, src_key_padding_mask=text_mask)
            enc_alignments.append(enc_align.unsqueeze(1))
        enc_alignments = torch.cat(enc_alignments, 1)

        ### Transformer Decoder ###
        tgt = decoder_input
        dec_alignments, enc_dec_alignments = [], []
        for layer in self.Decoder:
            tgt, dec_align, enc_dec_align = layer(tgt,
                                                  memory,
                                                  tgt_mask=diag_mask,
                                                  tgt_key_padding_mask=mel_mask,
                                                  memory_key_padding_mask=text_mask)
            dec_alignments.append(dec_align.unsqueeze(1))
            enc_dec_alignments.append(enc_dec_align.unsqueeze(1))
        dec_alignments = torch.cat(dec_alignments, 1)
        enc_dec_alignments = torch.cat(enc_dec_alignments, 1)

        ### Projection + PostNet ###
        mel_out = self.Projection(tgt.transpose(0, 1)).transpose(1, 2)
        mel_out_post = self.Postnet(mel_out) + mel_out

        gate_out = self.Stop(mel_out.transpose(1, 2)).squeeze(-1)
        
        return mel_out, mel_out_post, enc_alignments, dec_alignments, enc_dec_alignments, gate_out
    
    
    def forward(self, text, melspec, gate, text_lengths, mel_lengths, criterion):
        ### Size ###
        text = text[:,:text_lengths.max().item()]
        melspec = melspec[:,:,:mel_lengths.max().item()]
        gate = gate[:, :mel_lengths.max().item()]
        outputs = self.outputs(text, melspec, text_lengths, mel_lengths)
        
        mel_out, mel_out_post = outputs[0], outputs[1]
        enc_dec_alignments = outputs[4]
        gate_out=outputs[5]
        
        mel_loss, bce_loss, guide_loss = criterion((mel_out, mel_out_post, gate_out),
                                                   (melspec, gate),
                                                   (enc_dec_alignments, text_lengths, mel_lengths))
        
        return mel_loss, bce_loss, guide_loss
        
        
    def inference(self, text, max_len=1024):
        ### Size & Length ###
        (B, L), T = text.size(), max_len

        ### Prepare Inputs ###
        encoder_input = self.Embedding(text).transpose(0,1).contiguous()
        encoder_input += self.alpha1*(self.pe[:L].unsqueeze(1))

        ### Prepare Masks ###
        text_mask  = text.new_zeros(1, L).to(torch.bool)
        mel_mask = text.new_zeros(1, T).to(torch.bool)
        diag_mask = torch.triu(text.new_ones(T, T)).transpose(0, 1).contiguous()
        diag_mask[diag_mask == 0] = -1e9
        diag_mask[diag_mask == 1] = 0

        ### Transformer Encoder ###
        memory = encoder_input
        enc_alignments = []
        for layer in self.Encoder:
            memory, enc_align = layer(memory, src_key_padding_mask=text_mask)
            enc_alignments.append(enc_align)
        enc_alignments = torch.cat(enc_alignments, dim=0)

        ### Transformer Decoder ###
        mel_input = text.new_zeros(1,
                                   self.hp.n_mel_channels,
                                   max_len).to(torch.float32)
        dec_alignments = text.new_zeros(self.hp.n_layers,
                                        self.hp.n_heads,
                                        max_len,
                                        max_len).to(torch.float32)
        enc_dec_alignments = text.new_zeros(self.hp.n_layers,
                                            self.hp.n_heads,
                                            max_len,
                                            text.size(1)).to(torch.float32)

        ### Generation ###
        stop=[]
        for i in range(max_len):
            tgt = self.Prenet_D(mel_input.transpose(1,2).contiguous()).transpose(0,1).contiguous()
            tgt += self.alpha2*(self.pe[:T].unsqueeze(1))

            for j, layer in enumerate(self.Decoder):
                tgt, dec_align, enc_dec_align = layer(tgt,
                                                      memory,
                                                      tgt_mask=diag_mask,
                                                      tgt_key_padding_mask=mel_mask,
                                                      memory_key_padding_mask=text_mask)
                dec_alignments[j, :, i] = dec_align[0, :, i]
                enc_dec_alignments[j, :, i] = enc_dec_align[0, :, i]

            mel_out = self.Projection(tgt.transpose(0,1).contiguous())
            stop.append(torch.sigmoid(self.Stop(mel_out[:,i]))[0,0].item())

            if i < max_len - 1:
                mel_input[0, :, i+1] = mel_out[0, i]
                
            if stop[-1]>0.5:
                break

        mel_out_post = self.Postnet(mel_out.transpose(1, 2).contiguous())
        mel_out_post = mel_out_post.transpose(1, 2).contiguous() + mel_out
        mel_out_post = mel_out_post.transpose(1, 2).contiguous()

        return mel_out_post, enc_alignments, dec_alignments, enc_dec_alignments, stop

