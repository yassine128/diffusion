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

class UNet(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.time_emb = TimeEmbedding(emb_dim)
        self.class_emb = nn.Embedding(3, emb_dim)

        # encoder
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        self.res0 = ResBlock(64, 64, emb_dim)                       
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res1 = ResBlock(128, 128, emb_dim)                    
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.res2 = ResBlock(256, 256, emb_dim)                    
        self.down3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.res3 = ResBlock(256, 256, emb_dim)                    

        # middle
        self.mid1 = ResBlock(256, 256, emb_dim)
        self.mid2 = ResBlock(256, 256, emb_dim)

        # decoder
        self.res4 = ResBlock(256 + 256, 256, emb_dim)              
        self.up0 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(256, 256, 3, padding=1))
        self.res5 = ResBlock(256 + 256, 256, emb_dim)               
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(256, 256, 3, padding=1))
        self.res6 = ResBlock(256 + 128, 128, emb_dim)               
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(128, 128, 3, padding=1))
        self.res7 = ResBlock(128 + 64, 64, emb_dim)               

        # output
        self.norm_out = nn.GroupNorm(8, 64)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, t, y):
        emb = self.time_emb(t) + self.class_emb(y)

        h = self.conv_in(x)
        h = self.res0(h, emb);                 s0 = h 
        h = self.res1(self.down1(h), emb);     s1 = h 
        h = self.res2(self.down2(h), emb);     s2 = h
        h = self.res3(self.down3(h), emb);     s3 = h

        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        h = self.res4(torch.cat([h, s3], dim=1), emb)
        h = self.up0(h)                               
        h = self.res5(torch.cat([h, s2], dim=1), emb)
        h = self.up1(h)                                
        h = self.res6(torch.cat([h, s1], dim=1), emb)
        h = self.up2(h)                                  
        h = self.res7(torch.cat([h, s0], dim=1), emb)

        return self.conv_out(self.act(self.norm_out(h)))  

