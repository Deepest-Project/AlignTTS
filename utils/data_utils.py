import random
import numpy as np
import hparams
import torch
import torch.utils.data
import torch.nn.functional as F
import os
import pickle as pkl

from text import text_to_sequence

import matplotlib.pyplot as plt


def load_filepaths_and_text(metadata, split="|"):
    with open(metadata, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


class TextMelSet(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams, stage):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.data_type=hparams.data_type
        self.stage=stage
        
        self.text_dataset = []
        self.align_dataset = []
        seq_path = os.path.join(hparams.data_path, self.data_type)
        align_path = os.path.join(hparams.data_path, 'alignments')
        for data in self.audiopaths_and_text:
            file_name = data[0][:10]
            text = torch.from_numpy(np.load(f'{seq_path}/{file_name}_sequence.npy'))
            self.text_dataset.append(text)
            
            if stage !=0:
                align = torch.from_numpy(np.load(f'{align_path}/{file_name}_alignment.npy'))
                self.align_dataset.append(align)
            
            
    def get_mel_text_pair(self, index):
        file_name = self.audiopaths_and_text[index][0][:10]
        
        text = self.text_dataset[index]
        mel_path = os.path.join(hparams.data_path, 'melspectrogram')
        mel = torch.from_numpy(np.load(f'{mel_path}/{file_name}_melspectrogram.npy'))
        
        if self.stage == 0:
            return (text, mel)
        
        else:
            align = self.align_dataset[index]
            align = torch.repeat_interleave(torch.eye(len(align)).to(torch.long),
                                            align,
                                            dim=1)
            return (text, mel, align)

    def __getitem__(self, index):
        return self.get_mel_text_pair(index)

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    def __init__(self, stage):
        self.stage=stage
        return

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        max_target_len = max([x[1].size(1) for x in batch])
        num_mels = batch[0][1].size(0)

        if self.stage==0:
            text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
            mel_padded = torch.zeros(len(batch), num_mels, max_target_len)
            output_lengths = torch.LongTensor(len(batch))

            for i in range(len(ids_sorted_decreasing)):
                text = batch[ids_sorted_decreasing[i]][0]
                text_padded[i, :text.size(0)] = text
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, :mel.size(1)] = mel
                output_lengths[i] = mel.size(1)

            return text_padded, mel_padded, input_lengths, output_lengths
        
        
        else:
            text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
            mel_padded = torch.zeros(len(batch), num_mels, max_target_len)
            align_padded = torch.zeros(len(batch), max_input_len, max_target_len)
            output_lengths = torch.LongTensor(len(batch))

            for i in range(len(ids_sorted_decreasing)):
                text = batch[ids_sorted_decreasing[i]][0]
                text_padded[i, :text.size(0)] = text
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, :mel.size(1)] = mel
                output_lengths[i] = mel.size(1)
                align = batch[ids_sorted_decreasing[i]][2]
                align = align.T
                
#                 print(align.shape, align_padded.shape, text.shape, mel.shape)
#                 plt.figure()
#                 plt.imshow(align.cpu())
#                 plt.show()
                
                align_padded[i, :align.size(0), :align.size(1)] = align

            return text_padded, mel_padded, align_padded, input_lengths, output_lengths

    
    
    
    
    
    