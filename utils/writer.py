import os
from torch.utils.tensorboard import SummaryWriter
from .plot_image import *

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        raise Exception('The experiment already exists')
    else:
        os.mkdir(logging_path)
        writer = TTSWriter(logging_path)
            
    return writer


class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)
        
    def add_losses(self, mel_loss, bce_loss, guide_loss, global_step, phase):
        self.add_scalar(f'{phase}_mel_loss', mel_loss, global_step)
        self.add_scalar(f'{phase}_bce_loss', bce_loss, global_step)
        self.add_scalar(f'{phase}_guide_loss', guide_loss, global_step)
        
    def add_specs(self, mel_padded, mel_out, mel_out_post, mel_lengths, global_step, phase):
        mel_fig = plot_melspec(mel_padded, mel_out, mel_out_post, mel_lengths)
        self.add_figure(f'{phase}_melspec', mel_fig, global_step)
        
    def add_alignments(self, enc_alignments, dec_alignments, enc_dec_alignments,
                       text_padded, mel_lengths, text_lengths, global_step, phase):
        enc_align_fig = plot_alignments(enc_alignments, text_padded, mel_lengths, text_lengths, 'enc')
        self.add_figure(f'{phase}_enc_alignments', enc_align_fig, global_step)

        dec_align_fig = plot_alignments(dec_alignments, text_padded, mel_lengths, text_lengths, 'dec')
        self.add_figure(f'{phase}_dec_alignments', dec_align_fig, global_step)

        enc_dec_align_fig = plot_alignments(enc_dec_alignments, text_padded, mel_lengths, text_lengths, 'enc_dec')
        self.add_figure(f'{phase}_enc_dec_alignments', enc_dec_align_fig, global_step)
        
    def add_gates(self, gate_out, global_step, phase):
        gate_fig = plot_gate(gate_out)
        self.add_figure(f'{phase}_gate_out', gate_fig, global_step)