import torch
from torch.utils.data.dataset import Dataset
import os

class NetDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.data = self.read_data
        # Initialize
        # ...
        # ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # e.g. numpy -> tensor, data augmentation...
        # ...
        # ...
        pass

    @property
    def read_data(self):
        files = os.listdir(self.folder)
        print("Reading data... ", end='')

        data = []
        total = 0
        for file in files:
            # Preprocess file, then append to data
            # ...
            # ...
            total += 1

        print(f"Complete! Total: {total}")
        return data
