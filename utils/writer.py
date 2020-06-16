import os
from torch.utils.tensorboard import SummaryWriter
from .plot_image import *

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        writer = TTSWriter(logging_path)
        #raise Exception('The experiment already exists')
    else:
        os.mkdir(logging_path)
        writer = TTSWriter(logging_path)
            
    return writer


class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)
        
    def add_losses(self, mdn_loss, global_step, phase):
        self.add_scalar(f'{phase}_mdn_loss', mdn_loss, global_step)
        
    def add_specs(self, mel_padded, mel_out, mel_lengths, global_step, phase):
        mel_fig = plot_melspec(mel_padded, mel_out, mel_lengths)
        self.add_figure(f'{phase}_melspec', mel_fig, global_step)
        
    def add_alignments(self, probable_path, text_lengths, mel_lengths, global_step, phase):
        L, T = text_lengths[-1], mel_lengths[-1]
        align = probable_path[-1].contiguous()
        
        fig = plt.plot(figsize=(20,10))
        fig.imshow(align[:L, :T], origin='lower', aspect='auto')
        fig.xaxis.tick_top()
        self.add_figure(f'{phase}_enc_alignments', fig, global_step)