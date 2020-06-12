import torch
import torch.nn as nn
import torch.nn.functional as F
import hparams as hp
from utils.utils import get_mask_from_lengths
from torch.distributions import MultivariateNormal
import math

class MDNLoss(nn.Module):
    def __init__(self):
        super(MDNLoss, self).__init__()
        
    def forward(self, mu_sigma, melspec, text_lengths, mel_lengths):
        # mu, sigma: B, L, F / melspec: B, F, T
        B, L, _ = mu_sigma.size()
        T = melspec.size(2)
        
        x = melspec.transpose(1,2).unsqueeze(1) # B, 1, T, F
        mu = torch.sigmoid(mu_sigma[:, :, :hp.n_mel_channels].unsqueeze(2)) # B, L, 1, F
        sigma = torch.exp(mu_sigma[:, :, hp.n_mel_channels:].unsqueeze(2)) # B, L, 1, F
    
        exponential = -0.5*torch.sum((x-mu)*(x-mu)/sigma**2, dim=-1) # B, L, T
        coef = (2*math.pi)**(hp.n_mel_channels/2) * torch.prod(sigma, dim=-1)**0.5 # B, L, 1
        
        log_prob_matrix = exponential - torch.log(coef) # B, L, T
        log_alpha = mu_sigma.new_ones(B, L, T)*(-1e15)
        log_alpha[:,0, 0] = log_prob_matrix[:,0, 0]
        
        for t in range(1, T):
            prev_step = torch.cat([log_alpha[:, :, t-1:t], F.pad(log_alpha[:, :, t-1:t], (0,0,1,-1))], dim=-1)
            log_alpha[:, :, t] = torch.logsumexp(prev_step, dim=-1)+log_prob_matrix[:, :, t]
        
        alpha_last = torch.gather(log_alpha, -1, (mel_lengths-1).unsqueeze(-1).unsqueeze(-1).repeat(1, L, 1)).squeeze(-1)
        alpha_last = torch.gather(alpha_last, -1, (text_lengths-1).unsqueeze(-1))
        mdn_loss = -alpha_last.mean()

        return mdn_loss
        