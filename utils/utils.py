import hparams
from torch.utils.data import DataLoader
from .data_utils import TextMelSet, TextMelCollate
import torch
from text import *
import matplotlib.pyplot as plt


def prepare_dataloaders(hparams, stage):
    # Get data, data loaders and collate function ready
    trainset = TextMelSet(hparams.training_files, hparams, stage)
    valset = TextMelSet(hparams.validation_files, hparams, stage)
    collate_fn = TextMelCollate(stage)

    train_loader = DataLoader(trainset,
                              shuffle=True,
                              batch_size=hparams.batch_size, 
                              drop_last=True, 
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(valset,
                            batch_size=hparams.batch_size//hparams.n_gpus,
                            collate_fn=collate_fn)
    
    return train_loader, val_loader, collate_fn


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')

    
def lr_scheduling(opt, step, init_lr=hparams.lr, warmup_steps=hparams.warmup_steps):
    opt.param_groups[0]['lr'] = init_lr * min(step ** -0.5, step * warmup_steps ** -1.5)
    return


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len))
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    return mask


def reorder_batch(x, n_gpus, base=0):
    assert (x.size(0)%n_gpus)==0, 'Batch size must be a multiple of the number of GPUs.'
    base = base%n_gpus
    new_x = list(torch.zeros_like(x).chunk(n_gpus))
    for i in range(base, base+n_gpus):
        new_x[i%n_gpus] = x[i-base::n_gpus]
    
    new_x = torch.cat(new_x, dim=0)
    
    return new_x



def decode_text(padded_text, text_lengths, batch_idx=0):
    text = padded_text[batch_idx]
    text_len = text_lengths[batch_idx]
    text = ''.join([symbols[ci] for i, ci in enumerate(text) if i < text_len])
    
    return text
