import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset

split_sizes = {
    'train': 70000,
    'valid': 2500,
    'test': 2500
}

class DrawingDataset(Dataset):
    def __init__(self, data_path, split) -> None:
        if not os.path.isdir(data_path):
            raise ValueError(f"Provided data_path {data_path} isn't a directory. Do you have a typo?")
        self.data_path = data_path
        self.split = split
        if split not in split_sizes:
            raise ValueError(f"Incorrect split given. Supported {split_sizes.keys()}. Do you have a typo?")
        self.split_size = split_sizes[split]
        self.files = sorted(glob.glob(os.path.join(self.data_path, '*.npz')))
        
    def __len__(self):
        return len(self.files) * self.split_size

    def __getitem__(self, idx):
        file_path = idx // self.split_size
        index = idx % self.split_size
        
        label = os.path.basename(file_path).split('.')[0]
        data = np.load(file_path, encoding='latin1', allow_pickle=True)[self.split][index]
        
        return torch.from_numpy(data), label
        