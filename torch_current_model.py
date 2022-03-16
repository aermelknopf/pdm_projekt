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


class ReferenceModel(nn.Module):
    def __init__(self, nb_out, lstm_feat_size, lstm_hid_size, lstm_layers=2, lstm_batch_first=True):
        super().__init__()
        self.LSTM = torch.nn.LSTM(input_size=lstm_feat_size, hidden_size=lstm_hid_size, num_layers=lstm_layers, batch_first=lstm_batch_first)
        self.out = Linear(lstm_hid_size, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # print(f"input: {x.shape}")
        x, _ = self.LSTM(x)
        # print(f"lstm_output: {x.shape}")
        x = self.out(x[:, -1, :])
        # print(f"linear output: {x.shape}")
        x = self.out_activation(x)
        # print(f"output: {x.shape}")
        return x
