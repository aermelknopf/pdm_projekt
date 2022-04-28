# Slice LSTM implementation
# inspired by example from https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
# The LSTM-Cell uses the optimization of calculating the weights of all four LSTM gates in one big matrix
# (for input and hidden state units respective). Since gates are sliced, this results in a list of matrices.

import math
import torch
import torch.nn as nn


class SliceLSTM(nn.Module):
    """ Experimental new version of an LSTM layer, in which data slices of the input are processed by 'slices'
        of the typical LSTM gates (input gate i, forward gate f, g, output gate o) and TODO: complete """

    def __init__(self, lstm_slices: list[tuple]):
        super().__init__()
        # length of the input slices
        self.input_slices = [x[0] for x in lstm_slices]
        # length of the hidden units of the respective input slices (index-paired with self.input_slices)
        self.hidden_slices = [x[1] for x in lstm_slices]
        # hidden size of the entire layer
        self.hidden_size = sum(self.hidden_slices)
        # list of matrices of weights for input unit slices. Concatenation of weight matrix of all four gates
        # (i, f, g, o). One matrix contains weights for one slice of  i, f, g and o.
        self.Ws = nn.ParameterList([nn.Parameter(torch.Tensor(input_size, hidden_size * 4)) for input_size, hidden_size in lstm_slices])
        # list of matrices of weights for hidden unit slices. Concatenation of weight matrix of all four gates
        # (i, f, g, o). One matrix contains weights for one slice of  i, f, g and o.
        self.Us = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)) for _, hidden_size in lstm_slices])
        # list of bias vectors for each slice. One vector contains concatenated biases for all four gates i, f, g and o.
        self.biases = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_size * 4)) for _, hidden_size in lstm_slices])
        # weight matrix for the dense connector layer connecting the gate slices. Contains concatenated weights for all
        # four gates i, f, g and o.
        self.connector_Ws = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size * 4))
        # bias vector for the dense connector layer connecting the gate slices. Contains concatenated biases for all
        # four gates i, f, g and o.
        self.connector_biases = nn.Parameter(torch.Tensor(self.hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):

        # Some input dimension sanity checks:
        batch_size, sequence_length, feature_size = x.size()
        # print(f"input shape: {x.size()}")
        # print(f"batch size: {batch_size}, sequence_length: {sequence_length}, feature size: {feature_size}")
        total_input_size = sum(self.input_slices)
        assert(feature_size == total_input_size)

        # assumes x is of shape (batch, sequence, feature) !!!!
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        # TODO: other initialization possible
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            in_start = 0
            hid_start = 0
            slice_is, slice_fs, slice_gs, slice_os = [], [], [], []

            for index, (input_size, hidden_size) in enumerate(zip(self.input_slices, self.hidden_slices)):
                in_end = in_start + input_size          # end index exclusive in slicing
                hid_end = hid_start + hidden_size       # end index exclusive in slicing

                cur_x_t = x[:, t, in_start:in_end]      # input units for current slice (for entire batch)
                cur_h_t = h_t[:, hid_start:hid_end]     # hidden units for current slice (batch independent)

                # batch the computations for each slice into one single matrix multiplication
                gates = cur_x_t @ self.Ws[index] + cur_h_t @ self.Us[index] + self.biases[index]

                # apply activation functions for each gate slice
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :hidden_size]),                     # input gate slice
                    torch.sigmoid(gates[:, hidden_size: hidden_size * 2]),     # forget gate slice
                    torch.tanh(gates[:, hidden_size * 2: hidden_size * 3]),    # 'gate gate' slice
                    torch.sigmoid(gates[:, hidden_size * 3:]),                 # output gate slice
                )

                slice_is.append(i_t)
                slice_fs.append(f_t)
                slice_gs.append(g_t)
                slice_os.append(o_t)

            total_i = torch.cat(slice_is, dim=1)
            total_f = torch.cat(slice_fs, dim=1)
            total_g = torch.cat(slice_gs, dim=1)
            total_o = torch.cat(slice_os, dim=1)

            # apply dense connector-layer
            i_t = torch.sigmoid(total_i @ self.connector_Ws[:, :self.hidden_size]
                                + self.connector_biases[:self.hidden_size])
            f_t = torch.sigmoid(total_f @ self.connector_Ws[:, self.hidden_size:self.hidden_size * 2]
                                + self.connector_biases[self.hidden_size:self.hidden_size * 2])
            g_t = torch.tanh(total_g @ self.connector_Ws[:, self.hidden_size * 2:self.hidden_size * 3]
                             + self.connector_biases[self.hidden_size * 2:self.hidden_size * 3])
            o_t = torch.sigmoid(total_o @ self.connector_Ws[:, self.hidden_size * 3:]
                                + self.connector_biases[self.hidden_size * 3:])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()


        return hidden_seq, (h_t, c_t)
