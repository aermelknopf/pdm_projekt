import torch
import torch.nn as nn
from torch.nn import Linear

import torch_custom_lstm as custom
import custom_lstm_example as reference

class SliceModel(nn.Module):
    def __init__(self, nb_out):
        super().__init__()
        self.sliceLSTM1 = custom.SliceLSTM([(12, 20), (13, 20)])
        self.DropOut1 = nn.Dropout(p=0.2)
        self.sliceLSTM2 = custom.SliceLSTM([(20, 10), (20, 10)])
        self.DropOut2 = nn.Dropout(p=0.2)
        self.out = Linear(20, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.sliceLSTM1(x)
        x = self.DropOut1(x)
        x, _ = self.sliceLSTM2(x)
        x = x[:, -1, :]  # only take last hidden state (equals "return_sequence = False")
        x = self.DropOut2(x)
        x = self.out(x)
        x = self.out_activation(x)
        return x


class LibraryModel(nn.Module):
    def __init__(self, nb_out, lstm_feat_size, lstm_hid_size, lstm_layers=2, lstm_batch_first=True):
        super().__init__()
        self.LSTM = torch.nn.LSTM(input_size=lstm_feat_size, hidden_size=lstm_hid_size, num_layers=lstm_layers,
                                  batch_first=lstm_batch_first, dropout=0.2)
        self.out = Linear(lstm_hid_size, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # print(f"input: {x.shape}")
        x, _ = self.LSTM(x)
        x = x[:, -1, :]    # only take last output -> "return_sequence = False"
        # print(f"lstm_output: {x.shape}")
        x = self.out(x)
        # print(f"linear output: {x.shape}")
        x = self.out_activation(x)
        # print(f"output: {x.shape}")
        return x


class ReferenceCustomModel(nn.Module):
    def __init__(self, nb_out):
        super().__init__()
        self.LSTM1 = reference.CustomLSTM(25, 6)
        self.DropOut1 = nn.Dropout(p=0.2)
        self.LSTM2 = reference.CustomLSTM(6, 3)
        self.DropOut2 = nn.Dropout(p=0.2)
        self.out = Linear(3, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # print(f"input: {x.shape}")
        x, _ = self.LSTM1(x)
        x = self.DropOut1(x)
        x, _ = self.LSTM2(x)
        x = x[:, -1, :]  # only take last output -> "return_sequence = False"
        x = self.DropOut2(x)
        # print(f"lstm_output: {x.shape}")
        x = self.out(x)
        # print(f"linear output: {x.shape}")
        x = self.out_activation(x)
        # print(f"output: {x.shape}")
        return x
