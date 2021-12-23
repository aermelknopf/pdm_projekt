import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense


class SliceLSTM(keras.layers.layer):

    def __init__(self, lstm_slices, output_size, return_sequences=False):
        super(SliceLSTM, self).__init__()
        # some parameters which might be useful later
        self.num_slices = len(lstm_slices)
        self.output_size = output_size
        self.slice_inputs = [inputs for inputs, units in lstm_slices]
        self.slice_units = [units for inputs, units in lstm_slices]
        # nn components:
        self.LSTMs = [LSTM(units, return_sequences=return_sequences) for units in self.slice_units]
        self.connector = Dense(self.output_size)

    def call(self, input):
        x_slices = self.slice_input(input)
        y_slices = [lstm(input_slice) for lstm, input_slice in zip(self.LSTMs, x_slices)]
        #TODO: complete

    def slice_input(self, input):
        # assume input is of format: (batch , timesteps , feature)
        slices = []
        start_index = 0

        for input_size in self.slice_inputs:
            end_index = start_index + input_size
            slice = input_size[:, :, start_index:end_index]
            slices.append(slice)
            start_index = end_index

        return slices
