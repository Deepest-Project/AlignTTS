import os
from torch.utils.tensorboard import SummaryWriter
from .plot_image import *

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        writer = TTSWriter(logging_path)
#         raise Exception('The experiment already exists')
        print(f'The experiment {logging_path} already exists!')
    else:
        os.makedirs(logging_path)
        writer = TTSWriter(logging_path)
            
    return writer


class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)
        
    def add_specs(self, mel_target, mel_out, global_step, phase):
        fig, axes = plt.subplots(2, 1, figsize=(20,20))

        axes[0].imshow(mel_target,
                       origin='lower',
                       aspect='auto')

        axes[1].imshow(mel_out,
                       origin='lower',
                       aspect='auto')
    
        self.add_figure(f'{phase}_melspec', fig, global_step)
        
    def add_alignments(self, alignment, global_step, phase):
        fig = plt.plot(figsize=(20,10))
        plt.imshow(alignment, origin='lower', aspect='auto')
        self.add_figure(f'{phase}_alignments', fig, global_step)
        
        
