# -*- coding: utf-8 -*-
"""
Comparison of the network

Created on 2024

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""

import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from networks.attention import AttentionBlock


class TimeEmbedding(nn.Module):
    """
    Position embedding.
    """
    def __init__(self, T, d_model, dim):
        """

        :param T:           Timestep range
        :param d_model:     Dimension after position encoding
        :param dim:         Output dimension
        """
        assert d_model % 2 == 0                     # d_model must be an even number
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        """

        :param t:           Timestep
        :return:
        """
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    """
    Embedding for conditional information.
    """
    def __init__(self, num_class, d_model, dim):
        """

        :param num_class:   Category information range
        :param d_model:     Intermediate dimension
        :param dim:         Output dimension
        """
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_class, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, c):
        """

        :param c:           Category information vector
        :return:
        """
        emb = self.condEmbedding(c)
        return emb


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

    def forward(self, x, *args):
        """

        :param x:           Input feature map
        :param args:        Used to accept redundant inputs
        :return:
        """
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

    def forward(self, x, *args):
        """

        :param x:           Input feature map
        :param args:        Used to accept redundant inputs
        :return:
        """
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

        self.temb_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_ch * 2),
        )

        self.cond_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_ch),
        )

        self.zsem_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_ch),
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

    def forward(self, x, *args):
        """

        :param x:           Input feature map
        :param args:        Additional input information
                            (time series data, category information, semantic latent code)
        :return:
        """
        temb = args[0]
        cemb = args[1]
        zsem = args[2]

        # Project onto a fixed dimension
        temb = self.temb_proj(temb)[:, :, None, None]
        cemb = self.cond_proj(cemb)[:, :, None, None]
        zsem = self.cond_proj(zsem)[:, :, None, None]

        # AdaGN
        scale, shift = torch.chunk(temb, 2, dim=1)

        h = self.block1(x)
        h = shift + h * scale
        h = h * zsem

        h = self.block2(h)
        h += cemb
        h = self.block3(h)

        h = h + self.shortcut(x)
        h = self.attn(h)

        return h


class UNet(nn.Module):
    def __init__(self, T, n_class, ch, ch_mult, num_res_blocks, dropout):
        """

        :param T:               Timestep range
        :param n_class:         Category information range
        :param ch:              Basic number of channels
        :param ch_mult:         List about channel increase magnification
        :param num_res_blocks:  The number of residual blocks contained in each network unit
        :param dropout:         Drop out
        """
        super().__init__()

        tdim = ch * 4

        self.time_embedding = TimeEmbedding(T, ch, tdim)                    # Encoding for time step
        self.cond_embedding = ConditionalEmbedding(n_class, ch, tdim)       # Encoding for category information

        self.head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]                                                  # Record output channel when dowmsample for upsample

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

    def forward(self, x, *args):
        """

        :param x:           Input feature map
        :param args:        Additional input information
                            (time series data, category information, semantic latent code)
        :return:
        """

        t = args[0]                             # Time step vector
        n_class = args[1]                       # Category information vector
        zsem = args[2]                          # Semantic latent code

        temb = self.time_embedding(t)           # Time step embedding
        cemb = self.cond_embedding(n_class)     # Category information embedding

        # Down-sampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb, zsem)
            if isinstance(layer, ResBlock):
                hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb, zsem)

        # Up-sampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                shallow_fea = hs.pop()
                h = torch.cat([h, shallow_fea], dim=1)
            h = layer(h, temb, cemb, zsem)

        # End
        for layer in self.tail:
            if isinstance(layer, ResBlock):
                shallow_fea = hs.pop()
                h = torch.cat([h, shallow_fea], dim=1)
                h = layer(h, temb, cemb, zsem)
            else:
                h = layer(h)

        assert len(hs) == 0

        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000,
        n_class=3,
        ch=32,
        ch_mult=[1, 2, 3, 4],
        num_res_blocks=1,
        dropout=0.1)

    x = torch.randn(batch_size, 1, 64, 64)
    t = torch.randint(1000, (batch_size, ))
    zsem = torch.randn(batch_size, 128)
    n_class = torch.randint(3, (batch_size,))

    y = model(x, t, n_class, zsem)
    print(y.shape)
