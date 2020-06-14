from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

import os
import csv
import scipy.io.wavfile as wavfile

from jw_hparams import hparams

from g2p_en import G2p
# from text import *
from text import cmudict
from text.cleaners import custom_english_cleaners
from text.symbols import symbols

import librosa

Metadatum = namedtuple('metadatum', ('file_path', 'text'))

# Mappings from symbol to numeric ID and vice versa:
SYMBOL2ID = {s: i for i, s in enumerate(symbols)}
ID2SYMBOL = {i: s for i, s in enumerate(symbols)}

g2p = G2p()

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

def load_mel_spectrogram(wav_path, fs=None):
    '''
    return S # (n_mels, T)
    '''
    y, sr = load_wav_file(wav_path, fs)
    S = librosa.feature.melspectrogram(y, sr, n_mels=hparams['n_mels'], 
                                        n_fft=hparams['nsc'], 
                                        hop_length=hparams['hop'])

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

        return seq_tensor 

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

    print(hparams['metadata_path'])

    metadata_list = load_metadata(hparams['metadata_path'], hparams['wav_base_path'])

    dataset = PhoneMelDataset(metadata_list, hparams['fs'])

    data_loader = DataLoader(dataset,
                              num_workers=hparams['num_workers'],
                              shuffle=True,
                              batch_size=hparams['batch_size'], 
                              drop_last=False,
                              collate_fn=batch_collate_func)

    model = Model(hparams)

    for i, (seq, mel) in enumerate(data_loader):
        # print(seq, mel)

        tensor = model(seq, mel)

        print(tensor.shape)

        # load_wav_file(metadatum[0], 16000)

        if i == 10:
            break


    return

if __name__ == '__main__':


        
    main()