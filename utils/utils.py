import hparams
from torch.utils.data import DataLoader
from .data_utils import TextMelSet, TextMelCollate
import torch
from text import *
import matplotlib.pyplot as plt


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelSet(hparams.training_files, hparams)
    valset = TextMelSet(hparams.validation_files, hparams)
    collate_fn = TextMelCollate()

    train_loader = DataLoader(trainset,
                              num_workers=hparams.n_gpus-1,
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


def reorder_batch(x, n_gpus):
    assert (x.size(0)%n_gpus)==0, 'Batch size must be a multiple of the number of GPUs.'
    new_x = x.new_zeros(x.size())
    chunk_size = x.size(0)//n_gpus
    
    for i in range(n_gpus):
        new_x[i::n_gpus] = x[i*chunk_size:(i+1)*chunk_size]
    
    return new_x
