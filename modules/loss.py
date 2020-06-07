import torch
import torch.nn as nn
from utils.utils import get_mask_from_lengths


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        
    def forward(self, pred, target, guide):
        mel_out, mel_out_post, gate_out = pred
        mel_target, gate_target = target
        alignments, text_lengths, mel_lengths = guide
        
        mask = ~get_mask_from_lengths(mel_lengths)

        mel_target = mel_target.masked_select(mask.unsqueeze(1))
        mel_out_post = mel_out_post.masked_select(mask.unsqueeze(1))
        mel_out = mel_out.masked_select(mask.unsqueeze(1))

        gate_target = gate_target.masked_select(mask)
        gate_out = gate_out.masked_select(mask)
            
        mel_loss = nn.L1Loss()(mel_out, mel_target) + nn.L1Loss()(mel_out_post, mel_target)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))(gate_out, gate_target)
        guide_loss = self.guide_loss(alignments, text_lengths, mel_lengths)
        
        return mel_loss, bce_loss, guide_loss
    

    def guide_loss(self, alignments, text_lengths, mel_lengths):
        B, n_layers, n_heads, T, L = alignments.size()
        
        # B, T, L
        W = alignments.new_zeros(B, T, L)
        mask = alignments.new_zeros(B, T, L)
        
        for i, (t, l) in enumerate(zip(mel_lengths, text_lengths)):
            mel_seq = alignments.new_tensor( torch.arange(t).to(torch.float32).unsqueeze(-1)/t )
            text_seq = alignments.new_tensor( torch.arange(l).to(torch.float32).unsqueeze(0)/l )
            x = torch.pow(mel_seq-text_seq, 2)
            W[i, :t, :l] += alignments.new_tensor(1-torch.exp(-3.125*x))
            mask[i, :t, :l] = 1
        
        # Apply guided_loss to 2 heads of the last 2 layers 
        applied_align = alignments[:, -2:, :2]
        losses = applied_align*(W.unsqueeze(1).unsqueeze(1))
        
        return torch.mean(losses.masked_select(mask.unsqueeze(1).unsqueeze(1).to(torch.bool)))
