from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data import DrawingDataset

train_dataset = DrawingDataset(data_path="./data", split="train", max_length=100)
val_dataset = DrawingDataset(data_path="./data", split="valid", max_length=100)
test_dataset = DrawingDataset(data_path="./data", split="test", max_length=100)

train = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
test = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

print(f"Length of train dataloader: {len(train)}")
print(f"Length of validation dataloader: {len(val)}")
print(f"Length of test dataloader: {len(test)}")

for i,data in tqdm(enumerate(train)):
    sketch, label = data
    if i % 100000:
        print(sketch.shape, len(label))