from torchvision import datasets, transforms as T
from torch.utils.data import ConcatDataset, DataLoader
from helper import destroy
import matplotlib.pyplot as plt

tf = T.Compose([T.Resize(64), T.CenterCrop(64), T.RandomHorizontalFlip(),
                T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

ds = ConcatDataset([datasets.OxfordIIITPet("data", split=s, target_types="binary-category",
                    transform=tf, download=True) for s in ("trainval", "test")])

loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

x0, y = ds[0]
x0 = x0.unsqueeze(0)
x0, y = next(iter(loader))

xt = destroy(x0, 10)
img = ((xt[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0)
plt.imshow(img)
plt.show()