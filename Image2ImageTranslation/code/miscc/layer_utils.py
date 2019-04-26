"""Utility functions defined for specific blocks used in the architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from miscc.config import cfg

class Piling(nn.Module):
    """
        Spatial Piling the 2-dim tensor: input_var to target size
    """
    def __init__(self, target_size):
        super(Piling, self).__init__()
        self.target_size = target_size

    def forward(self, input_var):
        input_size = input_var.size() 
        input_var = input_var.unsqueeze(dim=2).unsqueeze(dim=3)
        output = input_var.expand(input_size[0], input_size[1], 
                self.target_size[0], self.target_size[1])

        return output

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        size_0 = x.size(0)
        return x.view(size_0, -1)

class SumPooling(nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, input_var):
        size = input_var.size()
        output = torch.sum(input_var.view(size[0], size[1], -1),
                dim=2)
        return output

class Identity(nn.Module):
    def __init__(self, *values):
        super(Identity, self).__init__()

    def forward(self, input_var):
        return input_var
