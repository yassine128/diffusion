import argparse
import copy
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_loader
from helper import destroy
from unet import UNet

T_STEPS = 1000
NO_CLASS = 2          
P_DROP = 0.1          
EMA_DECAY = 0.999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    loader = get_loader(args.batch_size, args.workers)
    model = UNet().to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)  
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    start_epoch = 0
    if args.resume and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        print("resumed from epoch", start_epoch)

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, n = 0.0, 0
        for x0, y in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x0, y = x0.to(device), y.to(device)
            B = x0.shape[0]

            t = torch.randint(0, T_STEPS, (B,), device=device)      
            xt, noise = destroy(x0, t)                               
            y = torch.where(torch.rand(B, device=device) < P_DROP,
                            torch.full_like(y, NO_CLASS), y)        

            loss = F.mse_loss(model(xt, t, y), noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():                                   
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA_DECAY).add_(pm.detach(), alpha=1 - EMA_DECAY)

            total += loss.item() * B
            n += B

        print(f"epoch {epoch + 1}: loss {total / n:.4f}")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "epoch": epoch}, args.ckpt)


if __name__ == "__main__":
    main()
