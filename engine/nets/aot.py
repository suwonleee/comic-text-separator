# GPL-3.0 — AOT inpainting generator
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def relu_nf(x):
    return F.relu(x) * 1.7139588594436646

def gelu_nf(x):
    return F.gelu(x) * 1.7015043497085571

def silu_nf(x):
    return F.silu(x) * 1.7881293296813965


class LambdaLayer(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class ScaledWSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 gain=True, eps=1e-4):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(self.eps).to(var.device)
        )) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class ScaledWSTransposeConv2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, gain=True, eps=1e-4):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size,
                                     stride, padding, output_padding, groups, bias,
                                     dilation, 'zeros')
        if gain:
            self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(self.eps).to(var.device)
        )) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x, output_size: Optional[List[int]] = None):
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose2d(x, self.get_weight(), self.bias, self.stride,
                                  self.padding, output_padding, self.groups, self.dilation)


class GatedWSConvPadded(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1, dilation=1):
        super().__init__()
        self.padding = nn.ReflectionPad2d(((ks - 1) * dilation) // 2)
        self.conv = ScaledWSConv2d(in_ch, out_ch, ks, stride=stride, dilation=dilation)
        self.conv_gate = ScaledWSConv2d(in_ch, out_ch, ks, stride=stride, dilation=dilation)

    def forward(self, x):
        x = self.padding(x)
        return self.conv(x) * torch.sigmoid(self.conv_gate(x)) * 1.8


class GatedWSTransposeConvPadded(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1):
        super().__init__()
        self.conv = ScaledWSTransposeConv2d(in_ch, out_ch, ks, stride=stride, padding=(ks - 1) // 2)
        self.conv_gate = ScaledWSTransposeConv2d(in_ch, out_ch, ks, stride=stride, padding=(ks - 1) // 2)

    def forward(self, x):
        return self.conv(x) * torch.sigmoid(self.conv_gate(x)) * 1.8


class ResBlock(nn.Module):
    def __init__(self, ch, alpha=0.2, beta=1.0, dilation=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.c1 = GatedWSConvPadded(ch, ch, 3, dilation=dilation)
        self.c2 = GatedWSConvPadded(ch, ch, 3, dilation=dilation)

    def forward(self, x):
        skip = x
        x = self.c1(relu_nf(x / self.beta))
        x = self.c2(relu_nf(x))
        return x * self.alpha + skip


def _layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    return 5 * (2 * (feat - mean) / std - 1)


class AOTBlock(nn.Module):
    def __init__(self, dim, rates=None):
        super().__init__()
        if rates is None:
            rates = [2, 4, 8, 16]
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True),
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = self.fuse(torch.cat(out, 1))
        mask = torch.sigmoid(_layer_norm(self.gate(x)))
        return x * (1 - mask) + out * mask


class AOTGenerator(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, ch=32, alpha=0.0):
        super().__init__()
        self.head = nn.Sequential(
            GatedWSConvPadded(in_ch, ch, 3, stride=1), LambdaLayer(relu_nf),
            GatedWSConvPadded(ch, ch * 2, 4, stride=2), LambdaLayer(relu_nf),
            GatedWSConvPadded(ch * 2, ch * 4, 4, stride=2),
        )
        self.body_conv = nn.Sequential(*[AOTBlock(ch * 4) for _ in range(10)])
        self.tail = nn.Sequential(
            GatedWSConvPadded(ch * 4, ch * 4, 3, 1), LambdaLayer(relu_nf),
            GatedWSConvPadded(ch * 4, ch * 4, 3, 1), LambdaLayer(relu_nf),
            GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2), LambdaLayer(relu_nf),
            GatedWSTransposeConvPadded(ch * 2, ch, 4, 2), LambdaLayer(relu_nf),
            GatedWSConvPadded(ch, out_ch, 3, stride=1),
        )

    def forward(self, img, mask):
        x = torch.cat([mask, img], dim=1)
        x = self.head(x)
        x = self.body_conv(x)
        x = self.tail(x)
        return x if self.training else torch.clip(x, -1, 1)
