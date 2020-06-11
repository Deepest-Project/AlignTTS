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
        sigma = torch.sigmoid(mu_sigma[:, :, hp.n_mel_channels:]).unsqueeze(2) # B, L, 1, F
        return mu, sigma
    
        exponential = -0.5*torch.sum((x-mu)*(x-mu)/sigma**2, dim=-1) # B, L, T
        coef = (2*math.pi)**(hp.n_mel_channels/2) * torch.prod(sigma, dim=-1)**0.5 # B, L, 1
        
        prob_matrix = torch.exp(exponential) / coef # B, L, T
        alpha = mu_sigma.new_zeros(B, L, T)
        alpha[:,0, 0] = prob_matrix[:,0, 0]
        
        for t in range(1, T):
            alpha[:, :, t] = (alpha[:, :, t-1] + F.pad(alpha[:, :, t-1], (1,-1))) * prob_matrix[:, :, t]
        
        alpha_last = torch.gather(alpha, -1, mel_lengths.unsqueeze(-1).unsqueeze(-1).repeat(1, L, 1)).squeeze(-1)
        alpha_last = torch.gather(alpha_last, -1, text_lengths.unsqueeze(-1))
        mdn_loss = -torch.log(alpha_last+1e-9).mean()

        return mdn_loss
        