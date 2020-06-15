from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence
from torch import optim

import os
import csv
import scipy.io.wavfile as wavfile
import math

from jw_hparams import hparams

from g2p_en import G2p
# from text import *
from text import cmudict
from text.cleaners import custom_english_cleaners
from text.symbols import symbols

import librosa
import numpy as np
from scipy.signal import stft

Metadatum = namedtuple('metadatum', ('file_path', 'text'))

# Mappings from symbol to numeric ID and vice versa:
SYMBOL2ID = {s: i for i, s in enumerate(symbols)}
ID2SYMBOL = {i: s for i, s in enumerate(symbols)}

g2p = G2p()

MEL_BANDS = librosa.filters.mel(sr=hparams['fs'], n_fft=hparams['nsc'], 
                                n_mels = hparams['n_mels'])

def text2phoneme(text): 
    clean_char = custom_english_cleaners(text.rstrip())
    clean_phone = []
    for s in g2p(clean_char.lower()):
        if '@'+s in SYMBOL2ID:
            clean_phone.append('@'+s)
        else:
            clean_phone.append(s)

    return clean_phone

def phoneme2seq(phoneme):
    sequence = [SYMBOL2ID['^']]
    sequence.extend([SYMBOL2ID[c] for c in phoneme])
    sequence.append(SYMBOL2ID['~'])
    return sequence

class PhoneMelDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_list, fs):
        self.metadata_list = metadata_list
        self.fs = fs

    def __getitem__(self, idx):
        file_path, text = self.metadata_list[idx]
        phoneme = text2phoneme(text)
        seq = phoneme2seq(phoneme)
        mel = load_mel_spectrogram(file_path, self.fs)

        return (seq, mel)

    def __len__(self):
        return len(self.metadata_list)

def load_metadata(metadata_path, base_path=''):

    metadata_list = list()

    with open(metadata_path, 'r') as file:
        
        for i, line in enumerate(file):
            file, _, text = line.strip().split('|')
            metadatum = Metadatum(os.path.join(base_path, file) + '.wav', text)
            metadata_list.append(metadatum)
            # print(file, text)
            # print(text2phoneme(text))
            # if i == 30:
            #     break
            
    return metadata_list

def load_mel_spectrogram(wav_path, fs=None, mode='scipy'):
    '''
    return S # (n_mels, T)
    '''
    y, sr = load_wav_file(wav_path, fs)

    if mode == 'librosa':

        S = librosa.feature.melspectrogram(y, sr, n_mels=hparams['n_mels'], 
                                            n_fft=hparams['nsc'], 
                                            hop_length=hparams['hop'])

    elif mode == 'scipy':

        f, t, Sxx = stft(y, fs=sr, nperseg=hparams['nsc'], noverlap=hparams['nov'])
        Zxx = np.abs(Sxx)
        mel_spectrogram = 20 * np.log10(np.maximum(MEL_BANDS @ Zxx, hparams['eps']))
        norm_coef = - 20 * np.log10(hparams['eps'])
        S = (mel_spectrogram + norm_coef) / norm_coef

    return S

def batch_collate_func(batch):

    seq_tensor_list = list()
    mel_tensor_list = list()

    for (seq_list, mel) in batch:
        seq_tensor_list.append(torch.LongTensor(seq_list))
        mel_tensor_list.append(torch.Tensor(mel).T)

    batch_seq = pad_sequence(seq_tensor_list)
    batch_mel = pad_sequence(mel_tensor_list)

    # print(batch_seq.shape, batch_mel.shape)

    return (batch_seq, batch_mel) # ((T, B), (T, B, n_mels))

def load_wav_file(wav_path, fs=None):
    sr, y = wavfile.read(wav_path)
    y = y / 2 ** 15

    if fs and sr != fs:
        y, sr = librosa.core.load(wav_path, sr=fs)

    # print(sr, max(y), min(y))

    return y, sr

class Model(torch.nn.Module):

    def __init__(self, hparams):
        super(Model, self).__init__()
        self.embedding_layer = nn.Embedding(len(SYMBOL2ID), 
                                    hparams['embedding_dim'], padding_idx=0)

        self.fft_layers = nn.ModuleList([FFT(hparams) for i  in range(hparams['num_FFT_blocks'])])

        self.mdn_layers = nn.ModuleList([MDNLinearLayer(hparams, i)
                                        for i in range(hparams['num_MDN_layers'])])

    def forward(self, seq_batch, mel_batch):

        seq_tensor = self.embedding_layer(seq_batch) # (T, B) -> (T, B, H)

        for fft_layer in self.fft_layers:
            seq_tensor = fft_layer(seq_tensor) # (T, B, H) -> (T, B, H)

        seq_tensor = seq_tensor.permute(1, 0, 2) # (T, B, H) -> (B, L, H)

        for mdn_layer in self.mdn_layers:
            seq_tensor = mdn_layer(seq_tensor) # (B, L, H) -> (B, L, H)

        seq_tensor = torch.sigmoid(seq_tensor)

        # print(seq_tensor[0, 0, :])

        # assert False

        logp_matrix = get_logp_matrix(seq_tensor, mel_batch)

        # print(logp_matrix)

        # print(logp_matrix)

        mdn_loss = get_mdn_loss(logp_matrix)

        return mdn_loss 


def get_logp_matrix(seq_tensor, mel_tensor):

    '''
    seq_tensor (B, L, H = 2 * Mel)
    mel_tensor (T, B, Mel)
    logp (B, L, T)
    '''

    mu_tensor = seq_tensor[:, :, :hparams['n_mels']] # (B, L, Mel)
    # log_sig_tensor = seq_tensor[:, :, hparams['n_mels']:] # (B, L, Mel)
    # sig_sqr_reciprocal_tensor = torch.exp(-2 * log_sig_tensor) # (B, L, Mel)
    sig_tensor = seq_tensor[:, :, hparams['n_mels']:] # (B, L, Mel)
    log_sig_tensor = torch.log(sig_tensor) # (B, L, Mel)
    sig_sqr_reciprocal_tensor = torch.reciprocal(sig_tensor ** 2)

    # https://github.com/jaywalnut310/glow-tts/blob/master/models.py#L310

    # log[ 1/(sig * sqrt(2*pi)) * exp{ -(x-mu)**2/(2*sig**2) } ]
    # = -log(sig) - 0.5*log(2*pi) - 0.5*(x**2/sig**2) + 1*(x*mu/sig**2) - 0.5*(mu**2/sig**2)
    # = [         logp1         ] + [     logp4    ]  + [    logp3      ] + [     logp2      ]

    # (B, L, Mel) * (B, L, Mel) -> (B, L, 1)
    
    logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - log_sig_tensor, [2]).unsqueeze(-1)
    
    # (B, L, Mel) x (B, Mel, T) = (B, L, T)
    logp2 = torch.matmul(sig_sqr_reciprocal_tensor, -0.5 * (mel_tensor.permute(1, 2, 0) ** 2)) 
    
    # (B, L, Mel) x (B, Mel, T) = (B, L, T)
    logp3 = torch.matmul((mu_tensor * sig_sqr_reciprocal_tensor), mel_tensor.permute(1, 2, 0)) 

    # (B, L, Mel) * (B, L, Mel) -> (B, L, 1)
    logp4 = torch.sum(-0.5 * (mu_tensor ** 2) * sig_sqr_reciprocal_tensor, [2]).unsqueeze(-1)

    logp = logp1 + logp2 + logp3 + logp4 # (B, L, T)

    # print(logp1[0, :10, :10])

    # print(logp2[0, :10, :10])

    # print(logp3[0, :10, :10])

    # print(logp4[0, :10, :10])

    return logp # (B, L, T)

def get_mdn_loss(logp_matrix):

    '''
    input: logp_matrix (B, L, T)

    return mdn_loss (B)
    '''

    B, L, T = logp_matrix.shape

    # logp_matrix = F.softmax(logp_matrix, dim=-1)

    logp_matrix_ = F.pad(log_alpha, (0,0,1,-1), value=-1e15)]

    # print(L, T)

    # a_matrix = torch.zeros(logp_matrix.shape)

    device = logp_matrix.device

    a_matrix = torch.zeros((B, L, T), device=device)

    a_matrix[:, 0, 0] = logp_matrix[:, 0, 0]

        alpha_last = torch.gather(log_alpha, -1, (mel_lengths-1).unsqueeze(-1).unsqueeze(-1).repeat(1, L, 1)).squeeze(-1)
        alpha_last = torch.gather(alpha_last, -1, (text_lengths-1).unsqueeze(-1))
        mdn_loss = -alpha_last.mean()

    for j in range(1, T):
        prev_step = torch.stack([a_matrix[:, :, j-1:j], F.pad(a_matrix[:, :, j-1:j], (1,-1, 0, 0)], dim=-1)
        a_matrix[:, :, j] = logp_matrix[:, :, j] + torch.logsumexp(prev_step, dim=-1)



    '''
    a_matrix = [[torch.zeros(B, device=device) for j in range(T)] for i in range(L)]

    for j in range(T):
        for i in range(min(j+1, L)):
            if i == 0 and j == 0:
                a_matrix[i][j] = logp_matrix[:, i, j]
            elif i == 0:
                a_matrix[i][j] = logp_matrix[:, i, j] + (a_matrix[i][j-1])
            else:
                a_matrix[i][j] = logp_matrix[:, i, j] + torch.log(torch.exp(a_matrix[i][j-1]) + 
                                                                        torch.exp(a_matrix[i-1][j-1]))

                # if float("-inf") in torch.log(torch.exp(a_matrix[i][j-1]) + torch.exp(a_matrix[i-1][j-1])):
                # print(a_matrix[i][j-1])
                # print(torch.exp(a_matrix[i][j-1]))
                # print(a_matrix[i-1][j-1])
                # print(torch.exp(a_matrix[i-1][j-1]))
                # print(torch.log(torch.exp(a_matrix[i][j-1]) + torch.exp(a_matrix[i-1][j-1])))
                # assert False

            # print(a_matrix[i][j])
    '''

    mdn_loss = -1 * a_matrix[L-1][T-1]

    # print(mdn_loss.requires_grad)
    # print(mdn_loss.grad_fn)

    return mdn_loss # (B)


class MDNLinearLayer(torch.nn.Module):
    '''
    (B, L, H)

    return (B, L, H)
    '''
    def __init__(self, hparams, layer_order):
        super(MDNLinearLayer, self).__init__()

        if layer_order == 0:
            self.layer_dim = hparams['MDN_dim']
            self.linear = nn.Linear(hparams['FFT_dim'], self.layer_dim) # (N, *, H)
        elif layer_order == hparams['num_MDN_layers'] - 1:
            self.layer_dim = 2 * hparams['n_mels']
            self.linear = nn.Linear(hparams['MDN_dim'], self.layer_dim) # (N, *, H)
        else:
            self.layer_dim = hparams['MDN_dim']
            self.linear = nn.Linear(hparams['MDN_dim'], self.layer_dim)

        self.batch_norm = nn.BatchNorm1d(self.layer_dim) # (N, C, L)
        # self.relu = nn.ReLU()
        self.relu = nn.ReLU6()
        self.dropout = nn.Dropout(hparams['MDN_dropout_rate'])


    def forward(self, input_tensor):

        tensor = self.linear(input_tensor) # (N, L, H)

        tensor = tensor.permute(0, 2, 1) # (N, L, H) -> (N, H, L)

        tensor = self.batch_norm(tensor) # (N, H, L)

        tensor = self.relu(tensor) # (N, H, L)

        tensor = self.dropout(tensor) # (N, H, L)

        tensor = tensor.permute(0, 2, 1) # (N, H, L) -> (N, L, H)

        return tensor


class FFT(torch.nn.Module):

    '''
    (T, B, H)

    return (T, B, H)
    '''

    def __init__(self, hparams):
        super(FFT, self).__init__()

        '''
        https://pytorch.org/docs/stable/nn.html#torch.nn.MultiheadAttention.forward
        '''

        self.multihead_attention = nn.MultiheadAttention(hparams['embedding_dim'], 
            num_heads=hparams['num_heads'], dropout=0.0, bias=True)

        self.batch_norm = nn.BatchNorm1d(hparams['FFT_dim'])

        self.conv_1d = nn.Conv1d(hparams['FFT_dim'], hparams['FFT_dim'], 
            kernel_size = hparams['kernel_size'], padding = 1) # (N, C, L) -> (N, C, L)

        self.batch_norm_2 = nn.BatchNorm1d(hparams['FFT_dim'])

    def forward(self, input_tensor):

        tensor, attention = self.multihead_attention(input_tensor, input_tensor, input_tensor) # (L, N, E) -> (L, N, E)

        tensor = tensor + input_tensor # (T, B, H) + (T, B, H)

        tensor = tensor.permute(1, 2, 0) # (T, B, H) => (B, H, T)

        tensor_ = self.batch_norm(tensor) # (N, C, L)

        tensor = self.conv_1d(tensor_)

        tensor = tensor + tensor_

        tensor = self.batch_norm_2(tensor) # (N, C, L)

        tensor = tensor.permute(2, 0, 1) # (B, H, T) -> (T, B, H)

        return tensor

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.autograd.set_detect_anomaly(True)

    print(hparams['metadata_path'])

    metadata_list = load_metadata(hparams['metadata_path'], hparams['wav_base_path'])

    dataset = PhoneMelDataset(metadata_list, hparams['fs'])

    data_loader = DataLoader(dataset,
                              num_workers=hparams['num_workers'],
                              shuffle=True,
                              batch_size=hparams['batch_size'], 
                              drop_last=False,
                              collate_fn=batch_collate_func)

    model = Model(hparams).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for i, (seq, mel) in enumerate(data_loader):
        print(seq.shape, mel.shape)

        optimizer.zero_grad()

        loss_tensor = model(seq.to(device), mel.to(device))

        loss = sum(loss_tensor)

        loss.backward()
        # grad_norm = commons.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        # print(tensor)

        # load_wav_file(metadatum[0], 16000)

        if i == 10:
            break


    return

if __name__ == '__main__':


        
    main()