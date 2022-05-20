# Class for profiling tests on dummy data to reduce different function calls.

import torch
import torch.nn as nn
from torch.nn import Linear
from torch.optim import Adam
from torch.utils.data import TensorDataset

from torch_lstm_experiment import train
import torch_custom_lstm as custom
import custom_lstm_example as reference

# SliceModel used for profiling
class SliceModel(nn.Module):

    def __init__(self, nb_out):
        super().__init__()
        self.sliceLSTM1 = custom.SliceLSTM([(12, 1), (13, 1)])
        # self.DropOut1 = nn.Dropout(p=0.1)
        # self.sliceLSTM2 = custom.SliceLSTM([(20, 10), (20, 10)])
        # self.DropOut2 = nn.Dropout(p=0.2)
        self.out = Linear(2, nb_out)
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.sliceLSTM1(x)
        # x = self.DropOut1(x)
        # x, _ = self.sliceLSTM2(x)
        x = x[:, -1, :]  # only take last hidden state (equals "return_sequence = False")
        # x = self.DropOut2(x)
        x = self.out(x)
        x = self.out_activation(x)
        return x


# ReferenceModel used for profiling
class ReferenceCustomModel(nn.Module):

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


if __name__ == '__main__':
    # STEP 1: create dummy data
    data_length = 100.000
    val_share = 0.1

    train_length = int(data_length * (1 - val_share))
    val_length = int(data_length * val_share)

    train_data = torch.rand((train_length, 50, 25))
    val_data = torch.rand((val_length, 50, 25))

    train_labels = torch.randint(0, 2, (train_length, 1), dtype=torch.float)    # random ints within [0,1]
    val_labels = torch.randint(0, 2, (val_length, 1), dtype=torch.float)        # random ints within [0,1]

    train_set = TensorDataset(train_data, train_labels)
    val_set = TensorDataset(val_data, val_labels)


    # STEP 2: create model
    model = SliceModel(1)
    learning_rate = 0.005


    # STEP 3: train model
    history = train(model=model, train_set=train_set, batch_size=1000, train_workers=4,
                    loss_fn=nn.BCELoss(), optimizer=Adam(model.parameters(), lr=learning_rate),
                    val_set=val_set, val_workers=4, n_epochs=50)
