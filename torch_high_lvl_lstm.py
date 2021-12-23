import torch
import torch.nn as nn


class SplitLSTM(nn.Module):
    def __init__(self, lstm_splits):
        super().__init__()
        self.hidden_size = sum(hid for (_, hid) in lstm_splits)
        self.input_splits = [inp for (inp, _) in lstm_splits]
        # self.hidden_splits = [hid for (_, hid) in lstm_splits]
        # self.split_sizes = [i for i in lstm_splits]
        self.num_splits = len(lstm_splits)

        split_fs = [nn.Linear(inp, hid) for (inp, hid) in lstm_splits]
        split_is = [nn.Linear((inp + hid), hid) for (inp, hid) in lstm_splits]
        split_gs = [nn.Linear((inp + hid), hid) for (inp, hid) in lstm_splits]
        split_os = [nn.Linear((inp + hid), hid) for (inp, hid) in lstm_splits]

        self.forget_gates = nn.ModuleList(split_fs)
        self.input_gates = nn.ModuleList(split_is)
        self.gs = nn.ModuleList(split_gs)
        self.output_gates = nn.ModuleList(split_os)

        connectors = [nn.Linear(self.hidden_size, self.hidden_size) for i in range(4)]
        self.connectors = nn.ModuleList(connectors)

        # self.slices = nn.ModuleList(slices)

    def forward(self, x, init_states = None):
        """ Assumes x to be of shape (batch_sz, sequence_sz, feature_sz) """
        batch_sz, seq_sz, _ = x.size()

        hidden_seq = []

        # initialize init_state
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_sz, self.hidden_size).to(x.device),
                        torch.zeros(batch_sz, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        # loop over sequence for recurrent training
        for t in range(seq_sz):
            split_outputs = []

            start_index = 0

            for i in range(self.num_splits):
                end_index = start_index + self.input_splits[i]
                x = x[:, t, start_index: end_index]

                split_f = self.split_fs[i](x)
                split_i = self.split_is[i](x)
                split_g = self.split_gs[i](x)
                split_o = self.split_os[i](x)

                # TODO: calc split outputs

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

