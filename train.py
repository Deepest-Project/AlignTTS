import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import Model
from modules.loss import TransformerLoss
import hparams
from text import *
from utils.utils import *
from utils.writer import get_writer


def validate(model, criterion, val_loader, iteration, writer):
    model.eval()
    with torch.no_grad():
        n_data, val_loss = 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            text_padded, text_lengths, mel_padded, mel_lengths, gate_padded = [
                x.cuda() for x in batch
            ]
            
            mel_out, mel_out_post,\
            enc_alignments, dec_alignments, enc_dec_alignments, gate_out = model.module.outputs(text_padded,
                                                                                                mel_padded,
                                                                                                text_lengths,
                                                                                                mel_lengths)
        
            mel_loss, bce_loss, guide_loss = criterion((mel_out, mel_out_post, gate_out),
                                                       (mel_padded, gate_padded),
                                                       (enc_dec_alignments, text_lengths, mel_lengths))
            
            loss = torch.mean(mel_loss+bce_loss+guide_loss)
            val_loss += loss.item() * len(batch[0])

        val_loss /= n_data

    writer.add_losses(mel_loss.item(),
                      bce_loss.item(),
                      guide_loss.item(),
                      iteration//hparams.accumulation, 'Validation')
    
    writer.add_specs(mel_padded.detach().cpu(),
                     mel_out.detach().cpu(),
                     mel_out_post.detach().cpu(),
                     mel_lengths.detach().cpu(),
                     iteration//hparams.accumulation, 'Validation')
    
    writer.add_alignments(enc_alignments.detach().cpu(),
                          dec_alignments.detach().cpu(),
                          enc_dec_alignments.detach().cpu(),
                          text_padded.detach().cpu(),
                          mel_lengths.detach().cpu(),
                          text_lengths.detach().cpu(),
                          iteration//hparams.accumulation, 'Validation')
    
    writer.add_gates(gate_out.detach().cpu(),
                    iteration//hparams.accumulation, 'Validation')
    model.train()
    
    
    
def main():
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams)
    model = nn.DataParallel(Model(hparams)).cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams.lr,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
    criterion = TransformerLoss()
    writer = get_writer(hparams.output_directory, hparams.log_directory)

    iteration, loss = 0, 0
    model.train()
    print("Training Start!!!")
    while iteration < (hparams.train_steps*hparams.accumulation):
        for i, batch in enumerate(train_loader):
            text_padded, text_lengths, mel_padded, mel_lengths, gate_padded = [
                reorder_batch(x, hparams.n_gpus).cuda() for x in batch
            ]

            mel_loss, bce_loss, guide_loss = model(text_padded,
                                                   mel_padded,
                                                   gate_padded,
                                                   text_lengths,
                                                   mel_lengths,
                                                   criterion)

            mel_loss, bce_loss, guide_loss=[
                torch.mean(x) for x in [mel_loss, bce_loss, guide_loss]
            ]
            sub_loss = (mel_loss+bce_loss+guide_loss)/hparams.accumulation
            sub_loss.backward()
            loss = loss+sub_loss.item()

            iteration += 1
            if iteration%hparams.accumulation == 0:
                lr_scheduling(optimizer, iteration//hparams.accumulation)
                nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()
                model.zero_grad()
                writer.add_losses(mel_loss.item(),
                                  bce_loss.item(),
                                  guide_loss.item(),
                                  iteration//hparams.accumulation, 'Train')
                loss=0


            if iteration%(hparams.iters_per_validation*hparams.accumulation)==0:
                validate(model, criterion, val_loader, iteration, writer)

            if iteration%(hparams.iters_per_checkpoint*hparams.accumulation)==0:
                save_checkpoint(model,
                                optimizer,
                                hparams.lr,
                                iteration//hparams.accumulation,
                                filepath=f'{hparams.output_directory}/{hparams.log_directory}')

            if iteration==(hparams.train_steps*hparams.accumulation):
                break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0,1')
    p.add_argument('-v', '--verbose', type=str, default='0')
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.verbose=='0':
        import warnings
        warnings.filterwarnings("ignore")
        
    main()