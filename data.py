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
        #self.file_paths = sorted(glob.glob(os.path.join(self.data_path, '*.npz')))
        
        # cat, sailboat, marker, headphones, airplane
        self.file_paths = ['./data/cat.npz', './data/sailboat.npz', './data/marker.npz', './data/headphones.npz', './data/airplane.npz']
        print(self.file_paths)
        
        prev_end = 0
        self.labels = P.IntervalDict()
        self.sketchs = []

        i = 0
        for path in tqdm(self.file_paths):
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
        scaled_sketch = self.sketchs[idx]
        data = torch.from_numpy(to_big_strokes(scaled_sketch, max_len=self.max_length))
        input_seq = data[:-1]
        target_seq = data[1:]
        return input_seq, target_seq
    
    def get_drawing_size(self, drawing):
        minx, miny = min(drawing[0, :]), min(drawing[1, :])
        maxx, maxy = max(drawing[0, :]), max(drawing[1, :])
        return int(maxx - minx), int(maxy - miny)

    def scale_drawing(self, drawing, scale):
        x,y = self.get_drawing_size(drawing)
        scalexy = scale / max(x, y, 1)
        drawing[0, :] = drawing[0, :] / scalexy
        drawing[1, :] = drawing[1, :] / scalexy
        return drawing

__all__ = ['DrawingDataset', 'collate_batch', 'split_sizes']