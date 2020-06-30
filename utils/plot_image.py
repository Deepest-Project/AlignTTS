from text import *
import torch
import hparams
import matplotlib.pyplot as plt


def plot_melspec(mel_target, mel_out):
    fig, axes = plt.subplots(2, 1, figsize=(20,20))

    axes[0].imshow(mel_target,
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(mel_out,
                   origin='lower',
                   aspect='auto')

    return fig


def plot_alignments(alignments, text, mel_lengths, text_lengths, att_type):
    fig, axes = plt.subplots(hparams.n_layers, hparams.n_heads, figsize=(5*hparams.n_heads,5*hparams.n_layers))
    L, T = text_lengths[-1], mel_lengths[-1]
    n_layers, n_heads = alignments.size(1), alignments.size(2)

    for layer in range(n_layers):
        for head in range(n_heads):
            if att_type=='enc':
                align = alignments[-1, layer, head].contiguous()
                axes[layer,head].imshow(align[:L, :L], aspect='auto')
                axes[layer,head].xaxis.tick_top()

            elif att_type=='dec':
                align = alignments[-1, layer, head].contiguous()
                axes[layer,head].imshow(align[:T, :T], aspect='auto')
                axes[layer,head].xaxis.tick_top()

            elif att_type=='enc_dec':
                align = alignments[-1, layer, head].transpose(0,1).contiguous()
                axes[layer,head].imshow(align[:L, :T], origin='lower', aspect='auto')
        
    return fig


def plot_gate(gate_out):
    fig = plt.figure(figsize=(10,5))
    plt.plot(torch.sigmoid(gate_out[-1]))
    return fig