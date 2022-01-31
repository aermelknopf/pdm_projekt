import torch
import torch.nn as nn
from torch.nn import Linear

import torch_custom_lstm as custom

class SliceModel(nn.Module):
    def __init__(self, nb_out):
        super().__init__()
        self.sliceLSTM1 = custom.SliceLSTM([(25, 4), (25, 4)], return_sequence=True)
        self.sliceLSTM2 = custom.SliceLSTM([(4, 2), (4, 2)], return_sequence=False)
        self.out = Linear(4, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.sliceLSTM1(x)
        x, _ = self.sliceLSTM2(x)
        x = self.out(x)
        x = self.out_activation(x)
        return x
