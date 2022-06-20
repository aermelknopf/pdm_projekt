# This file contains the last used LSTM models.
# It is used to reduce code changes in the torch_lstm_experiment.py file for clarity reasons. '''

import torch
import torch.nn as nn
from torch.nn import Linear

import torch_custom_lstm as custom
import custom_lstm_example as reference


class SlicedModel(nn.Module):
    """ Current version of the SlicedModel tested in torch_lstm_experiment.py """

    def __init__(self, nb_out):
        super().__init__()

        # first lstm_layer
        slices_1 = [(12, 4), (13, 4)]
        last_hidden_units = sum(item[1] for item in slices_1)
        self.sliceLSTM1 = custom.SlicedLSTM(slices_1)
        # TODO: check whether dropout is in forward!
        self.DropOut1 = nn.Dropout(p=0.20)

        # second lstm_layer
        # slices_2 = [(20, 10), (20, 10)]
        # last_hidden_units = sum(item[1] for item in slices_2)
        # self.sliceLSTM2 = custom.SliceLSTM(slices_2)
        # self.DropOut2 = nn.Dropout(p=0.2)

        # output layer
        self.out = Linear(last_hidden_units, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.sliceLSTM1(x)
        x = x[:, -1, :]  # only take last hidden state (equals "return_sequence = False")
        x = self.DropOut1(x)

        # x, _ = self.sliceLSTM2(x)
        # x = x[:, -1, :]  # only take last hidden state (equals "return_sequence = False")
        # x = self.DropOut2(x)
        x = self.out(x)
        x = self.out_activation(x)
        return x


class ReferenceCustomModel(nn.Module):
    """ Current version of the model used as comparison to SlicedModel in torch_lstm_experiment.py """

    def __init__(self, nb_out):
        super().__init__()
        self.LSTM1 = reference.CustomLSTM(25, 5)
        self.DropOut1 = nn.Dropout(p=0.1)
        # self.LSTM2 = reference.CustomLSTM(5, 2)
        # self.DropOut2 = nn.Dropout(p=0.1)
        self.out = Linear(5, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # print(f"input: {x.shape}")
        x, _ = self.LSTM1(x)
        x = self.DropOut1(x)
        # x, _ = self.LSTM2(x)
        x = x[:, -1, :]  # only take last output -> "return_sequence = False"
        # x = self.DropOut2(x)
        # print(f"lstm_output: {x.shape}")
        x = self.out(x)
        # print(f"linear output: {x.shape}")
        x = self.out_activation(x)
        # print(f"output: {x.shape}")
        return x


class LibraryModel(nn.Module):
    """ Current version of a fully integrated pytorch library LSTM model for ballpark comparison """

    def __init__(self, nb_out, lstm_feat_size, lstm_hid_size, lstm_layers=2, lstm_batch_first=True):
        super().__init__()
        self.LSTM = torch.nn.LSTM(input_size=lstm_feat_size, hidden_size=lstm_hid_size, num_layers=lstm_layers,
                                  batch_first=lstm_batch_first, dropout=0.2)
        self.out = Linear(lstm_hid_size, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # print(f"input: {x.shape}")
        x, _ = self.LSTM(x)
        x = x[:, -1, :]  # only take last output -> "return_sequence = False"
        # print(f"lstm_output: {x.shape}")
        x = self.out(x)
        # print(f"linear output: {x.shape}")
        x = self.out_activation(x)
        # print(f"output: {x.shape}")
        return x
