import matplotlib.pyplot as plt

from data import get_loader
from helper import destroy


def main():
    loader = get_loader(batch_size=64, num_workers=0)
    x0, y = next(iter(loader))
    print(x0.shape)                      # [64, 3, 64, 64]

    xt, noise = destroy(x0, 500)         # forward process, once, at t = 500

    img = ((xt[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
