import torch
from torch import nn
from dcnv4 import DCNv4  # Requires OpenGVLab's DCNv4 implementation


class DCNv4_Block(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1):
        super().__init__()
        self.dcn = DCNv4(
            in_channels=c1, out_channels=c2, kernel_size=k, stride=s, padding=p, group=g
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dcn(x)))