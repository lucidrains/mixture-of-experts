import torch
from torch import nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
