import os
import glob
import numpy as np
import portion as P
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


# Function from utils.py for sketch-rnn in the Magenta github repository
# at https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn
def to_big_strokes(stroke, max_len=100):
  """Converts from stroke-3 to stroke-5 format and pads to given length."""
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=np.float32)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result

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
        data = to_big_strokes(self.sketchs[idx], max_len=self.max_length)
        label = self.labels[idx]
        return torch.from_numpy(data), label
        