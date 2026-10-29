import argparse

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from helper import ALPHA, ALPHA_BAR, BETA
from unet import UNet

LABELS = {"cat": 0, "dog": 1, "none": 2}


@torch.no_grad()
def sample(model, label, n, w, device, seed=None):
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    alpha, alpha_bar, beta = ALPHA.to(device), ALPHA_BAR.to(device), BETA.to(device)
    x = torch.randn(n, 3, 64, 64, device=device, generator=gen)     
    y = torch.full((n,), label, device=device, dtype=torch.long)
    y_none = torch.full((n,), LABELS["none"], device=device, dtype=torch.long)

    for i in tqdm(reversed(range(len(beta))), total=len(beta), desc="sampling"):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        eps_c = model(x, t, y)                                       
        eps_u = model(x, t, y_none)                                  
        eps = eps_u + w * (eps_c - eps_u)                           

        mean = (x - beta[i] / torch.sqrt(1 - alpha_bar[i]) * eps) / torch.sqrt(alpha[i])
        if i > 0:
            x = mean + torch.sqrt(beta[i]) * torch.randn(x.shape, device=device, generator=gen)
        else:
            x = mean                                                 

    return ((x.clamp(-1, 1) + 1) / 2)                                


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=LABELS, default="cat")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--w", type=float, default=3.0, help="guidance strength, 1 = none")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    parser.add_argument("--out", default="samples.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["ema"])
    model.eval()

    imgs = sample(model, LABELS[args.label], args.n, args.w, device, args.seed)
    save_image(imgs, args.out, nrow=4)
    print("saved", args.out)


if __name__ == "__main__":
    main()
