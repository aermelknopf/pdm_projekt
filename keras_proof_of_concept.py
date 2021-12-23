# code from https://github.com/Azure-Samples/MachineLearningSamples-DeepLearningforPredictiveMaintenance/blob/master/Code/1_data_ingestion_and_preparation.ipynb



# import the libraries
import os
import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dropout, Dense, LSTM, Input, concatenate, Lambda
from keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler


import h5py

import matplotlib.pyplot as plt
import glob
import urllib


#  ~~~~ FUNCTION AREA ~~~~
# function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# function to read input files
def read_input_files():
    directory_string = 'CMaps/'
    files = ['FD001', 'FD002', 'FD003', 'FD004']
    filetype = '.txt'

    train_dfs = {}
    test_dfs = {}
    truth_dfs = {}

    for file in files:
        current_train_filepath = directory_string + "train_" + file + filetype
        current_test_filepath = directory_string + "test_" + file + filetype
        current_groundtruth_filepath = directory_string + "RUL_" + file + filetype

        train_df = pd.read_csv(current_train_filepath, sep=" ", header=None)
        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
        train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']

        # read test data
        test_df = pd.read_csv(current_test_filepath, sep=" ", header=None)
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
        test_df.columns = train_df.columns

        # read ground truth data
        truth_df = pd.read_csv(current_groundtruth_filepath, sep=" ", header=None)
        truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

        # Data Labeling - generate column RUL
        rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        train_df = train_df.merge(rul, on=['id'], how='left')
        train_df['RUL'] = train_df['max'] - train_df['cycle']
        train_df.drop('max', axis=1, inplace=True)

        train_dfs[file] = train_df
        test_dfs[file] = test_df
        truth_dfs[file] = truth_df

    return (train_dfs, test_dfs, truth_dfs)


# ~~~~ SCRIPT AREA ~~~~
# STEP -1: AGGREGATE AND READ DATA
data = read_input_files()




# STEP 0: CHOOSE DATA FROM DIRECTORY (changed from orig code: was azure blob stuff)
directory = "CMaps"
current_file = "all.txt"
current_train_filepath = directory + "/" + "train_" + current_file
current_test_filepath = directory + "/" + "test_" + current_file
current_groundtruth_filepath = directory + "/" + "RUL_" + current_file


# STEP 1: READING DATA
# read training data
train_df = pd.read_csv(current_train_filepath, sep=" ", header=None)
# train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read test data
test_df = pd.read_csv(current_test_filepath, sep=" ", header=None)
# test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = train_df.columns

# read ground truth data
truth_df = pd.read_csv(current_groundtruth_filepath, sep=" ", header=None)
# truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)


# STEP 2: DATA LABELING
# Data Labeling - generate column RUL
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

#train_df.to_csv('CMAPS/train_df_01', index=False)
#test_df.to_csv('CMAPS/test_df_01', index=False)

# generate label columns for training data
w1 = 30
w0 = 15

# Label1 indicates a failure will occur within the next 30 cycles.
# 1 indicates failure, 0 indicates healthy
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )

# label2 is multiclass, value 1 is identical to label1,
# value 2 indicates failure within 15 cycles
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2


# STEP 3: DATA NORMALIZATION (values between [0.0,1.0])
# MinMax normalization - train data
train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
min_max_scaler = MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)

# MinMax normalization - test data
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)


# STEP 4: GENERATE CLASS LABELS
# generate column max for test data
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)

# generate RUL for test data
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

# generate label columns w0 and w1 for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2


# FILE 2: MODEL BUILDING
# pick a large window size of 50 cycles
sequence_length = 50

# pick the feature columns
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
key_cols = ['id', 'cycle']
label_cols = ['label1', 'label2', 'RUL']

input_features = test_df.columns.values.tolist()
sensor_cols = [x for x in input_features if x not in set(key_cols)]
sensor_cols = [x for x in sensor_cols if x not in set(label_cols)]
sensor_cols = [x for x in sensor_cols if x not in set(sequence_cols)]

# The time is sequenced along
# This may be a silly way to get these column names, but it's relatively clear
sequence_cols.extend(sensor_cols)

# generator for the sequences
seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

# generate labels
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1'])
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)


# build the network
# Feature weights
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

# LSTM model (using keras sequential api)
model = Sequential()
model.add(LSTM(input_shape=(sequence_length, nb_features), units=10, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=3, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# plot_model(model, "sequential_model.png", show_shapes=True)


# LSTM model with keras functional api
input = Input(shape=(sequence_length, nb_features), name='input')
lstm_1 = LSTM(units=10, name='lstm_1', return_sequences=True)(input)
dropout_1 = Dropout(0.2, name='dropout_1')(lstm_1)
lstm_2 = LSTM(units=3, return_sequences=False, name='lstm_2')(dropout_1)
dropout_2 = Dropout(0.2, name='dropout_2')(lstm_2)
output = Dense(units=nb_out, activation='sigmoid', name='dense')(dropout_2)
functional_model = Model(inputs=input, outputs=output, name='functional_model')
functional_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# functional_model.summary()
# plot_model(functional_model, "functional_model.png", show_shapes=True)


# Split LSTM Model with keras functional api
input_ = Input(shape=(sequence_length, nb_features), name='input')
# left half
input_l = Lambda(lambda x: x[:, :25, :], name='split_l')(input_)
lstm_l1 = LSTM(units=4, return_sequences=True, name='lstm_l1')(input_l)
dropout_l1 = Dropout(0.2, name='dropout_l1')(lstm_l1)
lstm_l2 = LSTM(units=2, name='lstm_l2')(dropout_l1)
dropout_l2 = Dropout(0.2, name='dropout_l2')(lstm_l2)
# right half
input_r = Lambda(lambda x: x[:, 25:, :], name='split_r')(input_)
lstm_r1 = LSTM(units=4, return_sequences=True, name='lstm_r1')(input_r)
dropout_r1 = Dropout(0.2, name='dropout_r1')(lstm_r1)
lstm_r2 = LSTM(units=2, return_sequences=False, name='lstm_r2')(dropout_r1)
dropout_r2 = Dropout(0.2, name='dropout_r2')(lstm_r2)
# aggregate layer
concat = concatenate([dropout_l2, dropout_r2])
output_joint = Dense(units=nb_out, activation='sigmoid', name='output_joint')(concat)
split_model = Model(inputs=input_, outputs=output_joint, name='split_model')
split_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

split_model.summary()
plot_model(split_model, "graphs/split_model.png", show_shapes=True)


# fit the network
print("~~~~~~~~~ TRAINING SEQUENTIAL MODEL ~~~~~~~ ")
model.fit(seq_array, # Training features
           label_array, # Training labels
           epochs=10,   # We'll stop after 10 epochs
           batch_size=200, #
           validation_split=0.10, # Use 10% of data to evaluate the loss. (val_loss)
           verbose=1, #
           callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                      min_delta=0,    # until it doesn't change (or gets worse)
                                                      patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                      verbose=0,
                                                      mode='auto')])

print()
print()
print("~~~~~~~~~ TRAINING FUNCTIONAL MODEL ~~~~~~~ ")
functional_model.fit(seq_array, # Training features
           label_array, # Training labels
           epochs=10,   # We'll stop after 10 epochs
           batch_size=200, #
           validation_split=0.10, # Use 10% of data to evaluate the loss. (val_loss)
           verbose=1, #
           callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                      min_delta=0,    # until it doesn't change (or gets worse)
                                                      patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                      verbose=0,
                                                      mode='auto')])#

print()
print()
print("~~~~~~~~~ TRAINING SPLIT MODEL ~~~~~~~ ")
functional_model.fit(seq_array, # Training features
           label_array, # Training labels
           epochs=10,   # We'll stop after 10 epochs
           batch_size=200, #
           validation_split=0.10, # Use 10% of data to evaluate the loss. (val_loss)
           verbose=1, #
           callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                      min_delta=0,    # until it doesn't change (or gets worse)
                                                      patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                      verbose=0,
                                                      mode='auto')])#