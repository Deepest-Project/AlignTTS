import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 w_init_gain='linear'):
        super(Linear, self).__init__(in_dim,
                                     out_dim,
                                     bias)
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain(w_init_gain))


class Conv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 w_init_gain='linear'):
        super(Conv1d, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups,
                                     bias,
                                     padding_mode)
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain(w_init_gain))