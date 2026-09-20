import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.output_layer(x)
    

