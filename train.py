
import torch
from torch.utils.data import DataLoader

from data import DrawingDataset

train_dataset = DrawingDataset(data_path="./data", split="train")
val_dataset = DrawingDataset(data_path="./data", split="valid")
test_dataset = DrawingDataset(data_path="./data", split="test")

train = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
test = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)