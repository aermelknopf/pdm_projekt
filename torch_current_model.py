import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch_custom_lstm as custom

class SlicedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.slicedLSTM1 = custom.SliceLSTM()
        self.slicedLSTM2 = custm
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
