import os
import glob
import numpy as np
import portion as P
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from utils import *

split_sizes = {
    'train': 70000,
    'valid': 2500,
    'test': 2500
}

class DrawingDataset(Dataset):
    def __init__(self, data_path, split, max_length=250) -> None:
        if not os.path.isdir(data_path):
            raise ValueError(f"Provided data_path {data_path} isn't a directory. Do you have a typo?")
        self.data_path = data_path
        self.split = split
        if split not in set(('train', 'valid', 'test')):
            raise ValueError(f"Incorrect split given. Supported {split_sizes.keys()}. Do you have a typo?")
        self.max_length = max_length
        self.file_paths = sorted(glob.glob(os.path.join(self.data_path, '*.npz')))
        
        prev_end = 0
        self.labels = P.IntervalDict()
        self.sketchs = []
        num_files = 1#len(self.file_paths)
        i = 0
        for path in tqdm(self.file_paths):
            if i == num_files:
                break
            i += 1
            data = np.load(path, encoding='latin1', allow_pickle=True)[self.split]
            label = os.path.basename(path).split('.')[0]
            for sketch in data:
                if len(sketch) <= self.max_length:
                    self.sketchs.append(sketch)
            
            # Save label to interval
            self.labels[P.closed(prev_end, len(self.sketchs)-1)] = label
            prev_end = len(self.sketchs)
            
        
    def __len__(self):
        return len(self.sketchs)

    def __getitem__(self, idx):
        data = torch.from_numpy(to_big_strokes(self.sketchs[idx], max_len=self.max_length))
        # label = self.labels[idx]
        input_seq = data[:-1]
        target_seq = data[1:]
        return input_seq, target_seq
        
# def collate_batch(batch):
#     inputs, targets = zip(*batch)
#     inputs_pad = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
#     targets_pad = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
#     print(inputs_pad)
#     print(np.ndim(inputs_pad))
#     return inputs_pad, targets_pad

__all__ = ['DrawingDataset', 'collate_batch', 'split_sizes']