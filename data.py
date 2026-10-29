from torchvision import datasets, transforms as T
from torch.utils.data import ConcatDataset, DataLoader

def get_loader(batch_size=64, num_workers=0):
    tf = T.Compose([T.Resize(64), T.CenterCrop(64), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

    ds = ConcatDataset([datasets.OxfordIIITPet("data", split=s, target_types="binary-category",
                        transform=tf, download=True) for s in ("trainval", "test")])

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      persistent_workers=num_workers > 0, drop_last=True)
