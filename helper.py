import torch

BETA = torch.linspace(0.0001, 0.02, steps=1000)
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)

def destroy(x0, t=999):
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype)

    return torch.sqrt(ALPHA_BAR[t]) * x0 + torch.sqrt(1 - ALPHA_BAR[t]) * noise

    