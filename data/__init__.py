from data.triplet_input import TripleDataset
from data.image_input import ImageDataset
import torch.utils.data as data


class TripleDataLoader(data.Dataset):
    def __init__(self, opt):
        self.dataset = TripleDataset(opt.photo_root, opt.sketch_root)
        self.dataloader = data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=opt.batch_size,
            num_workers=4,
            drop_last=True
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __iter__(self):
        for i, data in enumerate(self.dataset):
            yield data


class ImageDataLoader(data.Dataset):
    def __init__(self, opt):
        self.dataset = ImageDataset(opt.image_root)
        self.dataloader = data.DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=opt.batch_size,
            num_workers=4,
            drop_last=False
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
    
    def __iter__(self):
        for i, data in enumerate(self.dataset):
            yield data