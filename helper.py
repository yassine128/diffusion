import torch

BETA = torch.linspace(0.0001, 0.02, steps=1000)
ALPHA = 1 - BETA
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)

def destroy(x0, t=999):
    noise = torch.randn_like(x0)
    t = torch.as_tensor(t, device=x0.device)
    ab = ALPHA_BAR.to(x0.device)[t].view(-1, 1, 1, 1) 

    xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    return xt, noise
