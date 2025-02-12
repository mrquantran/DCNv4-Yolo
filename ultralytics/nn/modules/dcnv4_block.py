import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

from ultralytics.nn.modules.conv import autopad


class DeformableConv2d(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, d=1, act=True):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            c1, 2 * k * k, k, s, autopad(k, p, d), dilation=d, bias=False
        )
        self.deform_conv = DeformConv2d(c1, c2, k, s, p)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return self.act(self.bn(out))
