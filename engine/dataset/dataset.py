import torch
import torch.utils.data as data

class DetDataset(data.Dataset):
    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        print()

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch

    def epoch(self):
        return self.epoch if hasattr(self, 'epoch') else -1
