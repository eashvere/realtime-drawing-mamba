from functools import partial
from torch import nn
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
import torch
class customModel(nn.Module):
    def __init__(self, nb, no, ns, embed_dim, drop_rate=0.1):
        super(customModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_features=5, out_features=self.embed_dim, bias=False)
        self.dropout = nn.Dropout1d(p=drop_rate)
        
        self.m1 = nn.ModuleList([create_block(self.embed_dim , device='cuda', layer_idx=f'm{i}') for i in range(nb)])
        self.leftm = nn.ModuleList([create_block(self.embed_dim , device='cuda', layer_idx=f'l{i}') for i in range(no)])
        self.rightm = nn.ModuleList([create_block(self.embed_dim , device='cuda', layer_idx=f'r{i}') for i in range(ns)])
        
        self.offset_hidden = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.state_hidden = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.relu = nn.ReLU()
        self.offset_out = nn.Linear(in_features=self.embed_dim, out_features=2)
        self.state_out = nn.Linear(in_features=self.embed_dim, out_features=3, bias=False)
        
        initializer_cfg = None
        for layer in self.m1:
            layer.apply(partial(_init_weights, n_layer=nb, **(initializer_cfg if initializer_cfg is not None else {})))
        for layer in self.leftm:
            layer.apply(partial(_init_weights, n_layer=no, **(initializer_cfg if initializer_cfg is not None else {})))
        for layer in self.rightm:
            layer.apply(partial(_init_weights, n_layer=ns, **(initializer_cfg if initializer_cfg is not None else {})))
        
            

    def forward(self, input): # x is of shape (B, L, 5) (Batchsize, sequence length, dimension)
        x = self.proj(input)
        x = self.dropout(x)
        hidden_states, residuals = x, None
        for layer in self.m1:
            hidden_states, residuals = layer(hidden_states, residuals)
        
        left_hidden_states, left_residuals = hidden_states, residuals
        right_hidden_states, right_residuals = hidden_states, residuals

        for layer in self.leftm:
            left_hidden_states, left_residuals = layer(left_hidden_states, left_residuals)
        for layer in self.rightm:
            right_hidden_states, right_residuals = layer(right_hidden_states, right_residuals)
        
        offset = self.relu(self.offset_hidden(left_hidden_states))
        offset = self.offset_out(offset)
        state = self.state_hidden(right_hidden_states)
        state = self.state_out(state)
        return offset, state

    @torch.inference_mode()
    def generate(model, input_seq):
        prev = input_seq
        
        i = 0
        while i < 20:
            offset, state = model.forward(prev)
            indices = torch.argmax(state, dim=-1)
            one_hot = torch.nn.functional.one_hot(indices, 3)
            pred = torch.cat((offset, one_hot), dim=-1)
            next_seg = pred[:, -1:, :]
            prev = torch.cat((prev, next_seg), dim=1)
            i += 1
        
        return prev

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
