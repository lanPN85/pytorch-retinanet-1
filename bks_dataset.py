from icdar_dataloader import IcdarDataset
from dataloader import Resizer, Normalizer, Augmenter
from torchvision import datasets, models, transforms


class UserDataset(IcdarDataset):
    def __init__(self, data_path):
        return super().__init__(data_path, 
            transform=transforms.Compose([
                Normalizer(), Augmenter(), Resizer()
            ])
        )


class UserValDataset(IcdarDataset):
    def __init__(self, data_path):
        return super().__init__(data_path, 
            transform=transforms.Compose([
                Normalizer(), Resizer()
            ])
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = UserDataset('data/icdar-task1-train')
    loader = DataLoader(ds, collate_fn=ds.collate)
    for r in loader:
        print(r)
