import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch_current_model as current_model


"""~~~~ FUNCTION AREA ~~~~ """
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

    return train_dfs, test_dfs, truth_dfs


# function to train the model
# inspired by: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(model, train_set, loss_fn, optimizer, batch_size=32, train_workers=2, val_workers=2, n_epochs=100, val_set=None, device='cpu'):
    # history dict for logging
    history = {'epoch': [], 'time': [], 'forward_time': [], 'backward_time': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=train_workers)
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=val_workers)
    else:
        val_loader = None

    size = len(train_loader.dataset)

    model.train()
    for i in range(n_epochs):
        correct = 0.0
        epoch_start_time = time.time()
        forward_duration = 0
        backward_duration = 0

        for batch, (X, y) in enumerate(train_loader):
            '''pass device variable to cuda and uncomment for gpu training
                change to tensors instead of numpy data first'''
            # X, y = X.to(device), y.to(device)

            start_time = time.time()
            # compute prediction error
            fwd_start_time = time.time()
            pred = model(X)              # assumes output shape to match label shape
            forward_duration += time.time() - fwd_start_time
            loss = loss_fn(pred, y)

            # backpropagation
            optimizer.zero_grad()
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_duration += time.time() - backward_start

            # calculate accuracy
            pred_classes = ((pred >= 0.5) == y)
            correct += pred_classes.sum().item()

            # measure time for epoch 'step' to print
            # stop_time = time.time()
            # elapsed_time = stop_time - start_time
            # timer += elapsed_time

            # print info multiple times per epoch
            # if batch > 0 and batch % 100 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>6d}/{size:>6d}]  time: {timer:.3f}s")
            #     timer = 0

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        loss = loss.item()
        accuracy = correct / size
        print(f"[Epoch {i}]     [{size:>6d}/{size:>6d}]")
        print(f"time: {epoch_duration:.3f}s  loss: {loss:>7f}  acc: {(100 * accuracy):>0.2f}%")

        history['epoch'].append(i)
        history['time'].append(round(epoch_duration, 4))
        history['forward_time'].append(round(forward_duration, 4))
        history['backward_time'].append(round(backward_duration, 4))
        history['loss'].append(round(loss, 6))
        history['accuracy'].append(round(accuracy, 6))

        if val_loader is not None:
            validate(model, val_loader, loss_fn, history=history)

    return history


# function for validation
def validate(model, dataloader, loss_fn, print_metrics=True, history=None, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            '''pass device variable and uncomment for gpu training'''
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # ACCURACY FORMULA WORKS ONLY FOR BINARY CLASSIFICATION
            # increase = (pred.argmax(1) == y).type(torch.float).sum().item() / X.shape[0]   <- ORIGINAL
            pred_classes = ((pred >= 0.5) == y)
            correct += pred_classes.sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    if print_metrics:
        print(f"Validation:    loss: {test_loss:>7f}  acc: {(100 * accuracy):>0.2f}%\n")

    if history is not None:
        if 'val_loss' in history.keys():
            history['val_loss'].append(round(test_loss, 6))

        if 'val_accuracy' in history.keys():
            history['val_accuracy'].append(round(accuracy, 6))


# write epoch_log analog to proof of concept function
def write_epoch_log(data_history, filepath, initial_string="epoch time fwd_time bwd_time loss acc val_loss val_acc\n"):
    output_lines = [initial_string]
    epochs = data_history['epoch']
    epoch_times = data_history['time']
    epoch_forward_times = data_history['forward_time']
    epoch_backward_times = data_history['backward_time']
    losses = data_history['loss']
    accs = data_history['accuracy']
    val_losses = data_history['val_loss']
    val_accs = data_history['val_accuracy']

    for (epoch, time, fwd_time, bwd_time, loss, acc, val_loss, val_acc) in zip(epochs, epoch_times, epoch_forward_times, epoch_backward_times, losses, accs, val_losses, val_accs):
        out_line = f"{epoch} {time:.4f} {fwd_time:.4f} {bwd_time:.4f} {loss:.6f} {acc:.6f} {val_loss:.6f} {val_acc:.6f}\n"
        output_lines.append(out_line)

    with open(filepath, "w") as file:
        file.writelines(output_lines)


''' ~~~~~~~~~~~~~~~~~~~~~
    ~~~~ SCRIPT AREA ~~~~ '''
if __name__ == '__main__':
    # STEP -1: AGGREGATE AND READ DATA
    # data = read_input_files()


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
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
    min_max_scaler = MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)

    # MinMax normalization - test data
    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize,
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns=test_df.columns)
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


    # FILE 2: MODEL BUILDING AND TRAINING
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

    # generate labels and convert to numpy array
    label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1'])
                 for id in train_df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)


    # evanuating class fequency

    samples = label_array.shape[0]
    print(f"samples: {samples}")
    one_samples = label_array.sum(axis=0)[0]
    one_frequency = one_samples / samples
    zero_samples = (samples - one_samples)
    zero_frequency = zero_samples / samples

    print(f"positive (class '1'): {int(one_samples)}, ({100*one_frequency:.2f}%)")
    print(f"negative (class '0'): {int(zero_samples)}, ({100*zero_frequency:.2f}%)")
