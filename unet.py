import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, sin_dim, emb_dim):
        super().__init__()
        self.sin_dim = sin_dim
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):
        half = self.sin_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        angles = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, heads=4, groups=32):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = (t.transpose(-1, -2) for t in qkv.unbind(1))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class Block(nn.Module):
    """ResBlock, optionally followed by self-attention."""

    def __init__(self, in_ch, out_ch, emb_dim, use_attn, groups=32):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim, groups)
        self.attn = AttentionBlock(out_ch, groups=groups) if use_attn else nn.Identity()

    def forward(self, x, emb):
        return self.attn(self.res(x, emb))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, base=128, mults=(1, 2, 4, 4), n_res=3, attn_res=(32, 16, 8),
                 img_size=64, n_classes=3, groups=32):
        super().__init__()
        emb_dim = base * 4
        self.time_emb = TimeEmbedding(base, emb_dim)
        self.class_emb = nn.Embedding(n_classes, emb_dim)
        self.conv_in = nn.Conv2d(3, base, 3, padding=1)

        self.down = nn.ModuleList()
        chs = [base]
        ch, res = base, img_size
        for i, m in enumerate(mults):
            out = base * m
            for _ in range(n_res):
                self.down.append(Block(ch, out, emb_dim, res in attn_res, groups))
                ch = out
                chs.append(ch)
            if i < len(mults) - 1:
                self.down.append(Downsample(ch))
                chs.append(ch)
                res //= 2

        self.mid1 = ResBlock(ch, ch, emb_dim, groups)
        self.mid_attn = AttentionBlock(ch, groups=groups)
        self.mid2 = ResBlock(ch, ch, emb_dim, groups)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            out = base * m
            for _ in range(n_res + 1):
                self.up.append(Block(ch + chs.pop(), out, emb_dim, res in attn_res, groups))
                ch = out
            if i > 0:
                self.up.append(Upsample(ch))
                res *= 2

        self.norm_out = nn.GroupNorm(groups, ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t, y):
        emb = self.time_emb(t) + self.class_emb(y)

        h = self.conv_in(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, Block) else layer(h)
            hs.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for layer in self.up:
            if isinstance(layer, Block):
                h = layer(torch.cat([h, hs.pop()], dim=1), emb)
            else:
                h = layer(h)

        return self.conv_out(F.silu(self.norm_out(h)))