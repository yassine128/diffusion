import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(64, 256) 
        self.act = nn.SiLU(inplace=False)
        self.lin2 = nn.Linear(256, 256)

    def sinusoidal_embedding(self, t, dim):
        half_dim = dim // 2

        frequencies = torch.exp(
            -math.log(10000) *
            torch.arange(half_dim, device=t.device) / half_dim
        )

        angles = t.float().reshape(-1, 1) * frequencies.reshape(1, -1)

        embedding = torch.cat([
            torch.sin(angles),
            torch.cos(angles)
        ], dim=1)

        return embedding

    def forward(self, t):
        emb = self.sinusoidal_embedding(t, 64)
        return self.lin2(self.act(self.lin1(emb)))

class labelEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.emb = nn.Embedding(3, dim)

    def forward(self, y):
        return self.emb(y)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_ch, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(in_features=emb_dim, out_features=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch, eps=1e-05, affine=True)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=in_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=False)
        self.skip = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=1) if in_ch != out_ch else nn.Identity

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        e = self.emb_proj(self.act(emb))
        h = h + e[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

