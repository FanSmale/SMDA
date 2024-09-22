# -*- coding: utf-8 -*-
"""
Comparison of the network

Created on 2024

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from networks.attention import AttentionBlock


class DownSample(nn.Module):
    """
    Down-sampling.
    """
    def __init__(self, in_ch):
        """

        :param in_ch:       Number of input channel
        """
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    """
    Up-sampling.
    """
    def __init__(self, in_ch):
        """

        :param in_ch:       Number of input channel
        """
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class ResBlock(nn.Module):
    """
    Residual structure.
    """
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        """

        :param in_ch:       Number of input channel
        :param out_ch:      Number of output channel
        :param tdim:        Number of intermediate channel
        :param dropout:     Drop out
        :param attn:        Is there an embedded attention mechanism
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
        )

        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
        )

        self.block3 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        """

        :param x:           Input feature map
        :return:
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)

        h = h + self.shortcut(x)
        h = self.attn(h)

        return h


class AutoEnc(nn.Module):
    def __init__(self, ch, ch_mult, num_res_blocks, dropout):
        """

        :param ch:              Basic number of channels
        :param ch_mult:         List about channel increase magnification
        :param num_res_blocks:  The number of residual blocks contained in each network unit
        :param dropout:         Drop out
        """
        super().__init__()

        tdim = ch * 4

        self.head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        self.semantic = None
        chs = [ch]                          # record output channel when dowmsample for upsample

        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            self.downblocks.append(AttentionBlock(now_ch))
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout),
            AttentionBlock(now_ch),
            ResBlock(now_ch, now_ch, tdim, dropout),
        ])

        self.semblocks1 = DownSample(now_ch)
        self.semblocks2 = nn.Conv2d(now_ch, now_ch, 3, stride=1, padding=1)
        self.semblocks3 = UpSample(now_ch)

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
            self.upblocks.append(AttentionBlock(now_ch))
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.tail = nn.Sequential(
            ResBlock(in_ch=chs.pop() + now_ch, out_ch=now_ch, tdim=tdim, dropout=dropout),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )

        assert len(chs) == 0
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x):
        """

        :param x:           Input feature map
        :return:            Decoder output
        """
        # Down-sampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h)
            if isinstance(layer, ResBlock):
                hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h)

        # Semantic
        h = self.semblocks1(h)
        self.semantic = self.semblocks2(h)
        h = self.semblocks3(self.semantic)

        # Up-sampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                shallow_fea = hs.pop()
                h = torch.cat([h, shallow_fea], dim=1)
            h = layer(h)

        # End
        for layer in self.tail:
            if isinstance(layer, ResBlock):
                shallow_fea = hs.pop()
                h = torch.cat([h, shallow_fea], dim=1)
                h = layer(h)
            else:
                h = layer(h)

        assert len(hs) == 0

        return h

    def encode(self, x):
        """

        :param x:           Input feature map
        :return:            Encoder output
        """

        # Down-sampling
        h = self.head(x)
        for layer in self.downblocks:
            h = layer(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h)

        # Semantic
        h = self.semblocks1(h)
        self.semantic = self.semblocks2(h)

        return torch.mean(torch.mean(self.semantic, dim=2), dim=2)


if __name__ == '__main__':
    batch_size = 8
    model = AutoEnc(ch=32, ch_mult=[1, 2, 3, 4], num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 1, 64, 64)

    print(model(x).shape)
    print(model.encode(x).size())
