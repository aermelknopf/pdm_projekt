import matplotlib.pyplot as plt
import os
import pandas
import numpy as np


def read_files(dir, selected=None, round_decimals=None, root_dir=None):
    dfs = {}

    for filename in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, filename)):  # filter subdirectories

            if file_interesting(filename, selection=selected):
                root_dir_string = "" if root_dir is None else f"{root_dir}/"

                df = pandas.read_csv(f"{root_dir_string}{dir}/{filename}", sep=" ")
                if round_decimals is not None:
                    df = df.round(decimals=round_decimals)

                dfs[filename] = df

    return dfs


def file_interesting(filename: str, selection=None):
    interesting = False

    if selection is None:
        interesting = True
    else:
        if callable(selection):
            interesting = selection(filename)
        elif filename in selection:
            interesting = True

    return interesting


def line_plot(df_dict, column, value_factor=1, aggregate=None, xlabel=None, ylabel=None, title=None, legend=None,
              show=False, savepath=None):
    plot_data = []

    for label, df in df_dict.items():
        xs = df['epoch'].values
        ys = df[column].values * value_factor

        # potentially change name of legend (default: file name)
        if legend is not None:
            if type(legend) is dict:
                if label in legend:
                    label = legend[label]
            elif callable(legend):
                label = legend(label)

        plot_data.append((label, xs, ys))

    if aggregate is not None:
        aggregated = aggregate_df_dict(df_dict, aggregate=aggregate)
        xs = aggregated['epoch'].values
        ys = aggregated[column].values * value_factor
        plot_data.append((aggregate, xs, ys))

    for (label, xs, ys) in plot_data:
        plt.plot(xs, ys, label=label)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if legend is not None:
        plt.legend()

    # needs to happen before plt.show() else only white image will be plotted
    if savepath is not None:
        plt.savefig(savepath)

    if show:
        plt.show()

    # figure needs to be cleared!
    plt.clf()


def plot_poc():
    result_dir = "results/proof of concept"

    selected = ["functional_model.txt", "functional_model2.txt", "functional_model3.txt",
                "split_model4.txt", "split_model5.txt", "split_model6.txt"]

    legend = {"functional_model.txt": "normal LSTM (run 1)", "functional_model2.txt": "normal LSTM (run 2)",
              "functional_model3.txt": "normal LSTM (run 3)", "split_model.txt": "time split LSTM (run 1)",
              "split_model2.txt": "time split LSTM (run 2)", "split_model3.txt": "time split LSTM (run 3)",
              "split_model4.txt": "split LSTM (run 1)", "split_model5.txt": "split LSTM (run 2)",
              "split_model6.txt": "split LSTM (run 3)"}

    dfs = read_files(result_dir, round_decimals=3, selected=selected)
    # dfs["functional_model.txt"]["val_acc"].plot()
    # plt.show()

    line_plot(dfs, column="time", xlabel="training epoch", ylabel="epoch training time [s]",
              legend=legend,
              title="Training Epoch Time of Normal and Split LSTM model",
              show=True, savepath="graphs/poc_training_time")

    line_plot(dfs, column="val_acc", value_factor=100, xlabel="training epoch", ylabel="validation accuracy [%]",
              legend=legend,
              title="Validation Accuracy of Normal and Split LSTM model",
              show=True, savepath="graphs/poc_val_acc")


def get_title_string(model_string: str):
    model_architecture = []
    while model_string != '':
        if model_string[0] == '(':
            layer_end_index = model_string.find(")")
            layer_string = model_string[1: layer_end_index]
            model_string = model_string[layer_end_index + 1:]

            layer_string, model_string = parse_sliced_layer(model_string)
        elif model_string[0] == 'd':
            layer_string, model_string = parse_dropout_layer(model_string)

        model_architecture.append(layer_string)


def parse_sliced_layer(model_string: str):
    # TODO: complete?
    pass


def parse_dropout_layer(model_string: str):
    # TODO: complete?
    pass


def plot_run_comparison(root_dir: str, learning_rates=(0.001, 0.005, 0.01, 0.02), aggregate=None, save=False,
                        show=True):
    model_type = root_dir.partition("/")[2]
    save_root_dir = os.path.join("graphs/run comparisons", model_type)

    for item in os.listdir(root_dir):  # iterate over direct subdirectories (representing one model architecture)
        cur_dir = os.path.join(root_dir, item)
        cur_save_dir = os.path.join(save_root_dir, item)

        if os.path.isdir(cur_dir):
            for lr in learning_rates:  # iterate over learning rates
                selector = get_lr_filter(lr)
                data = read_files(cur_dir, selected=selector)

                if bool(data):
                    title = f"{item}  lr={lr}"
                    legend = lambda s: "run " + s[-1]

                    if save:
                        os.makedirs(cur_save_dir, exist_ok=True)
                        filename = f"lr-{get_lr_string(lr)}"
                        if aggregate is not None:
                            filename += (f"-{aggregate}")
                        filename += (".png")
                        save_path = os.path.join(cur_save_dir, filename)
                    else:
                        save_path = None

                    line_plot(data, column="val_acc", show=show, aggregate=aggregate, legend=legend, title=title,
                              xlabel="training epoch", ylabel="validation accuracy [%]", value_factor=100,
                              savepath=save_path)


def get_lr_string(lr: float):
    lr_string = str(lr)
    lr_string = lr_string.partition('.')[2]  # substring after seperator parameter ('.')
    return lr_string


def get_lr_filter(lr):
    lr_string = get_lr_string(lr)
    lr_file_string = f"lr{lr_string}"
    return lambda s: lr_file_string in s


def aggregate_df_dict(dfs: dict, aggregate="mean"):
    if not dfs:
        ValueError("no dfs in dict to aggregate in aggregate_df_dict")

    df_list = [i for i in dfs.values()]
    concated = pandas.concat(df_list)
    grouped = concated.groupby(level=0)

    if aggregate == "mean":
        aggregated = grouped.mean()
    elif aggregate == "median":
        aggregated = grouped.median()
    elif callable(aggregate):
        aggregated = aggregate(dfs)
    else:
        ValueError("aggregate must be 'mean', 'median' or callable")
    return aggregated


def plot_lr_comparison(root_dir: str, aggregate="mean", learning_rates=(0.001, 0.005, 0.01, 0.02), save=False,
                       show=True):
    model_type = root_dir.partition("/")[2]
    save_root_dir = os.path.join("graphs/lr comparisons", model_type)

    for item in os.listdir(root_dir):  # iterate over direct subdirs (representing one model architecture)
        cur_dir = os.path.join(root_dir, item)
        cur_save_dir = os.path.join(save_root_dir, item)

        if os.path.isdir(cur_dir):
            aggregated_data = {}

            for lr in learning_rates:  # iterate over learning rates
                selector = get_lr_filter(lr)

                data = read_files(cur_dir, selected=selector)
                aggregated_data[str(lr)] = aggregate_df_dict(data, aggregate=aggregate)

            if bool(aggregated_data):
                title = f"{item}    {aggregate} of different learning rates"
                legend = lambda s: f"lr={s}"  # no augmentation should not be necessary

                if save:
                    os.makedirs(cur_save_dir, exist_ok=True)
                    filename = (f"{aggregate}.png")
                    save_path = os.path.join(cur_save_dir, filename)
                else:
                    save_path = None

                line_plot(aggregated_data, column="val_acc", show=show, aggregate=None, legend=legend, title=title,
                          xlabel="training epoch", ylabel="validation accuracy [%]", value_factor=100,
                          savepath=save_path)


def read_all_data(aggregate=None):
    learning_rates = [0.001, 0.005, 0.01, 0.02]
    root_dir = 'results'
    dir_filter = ('reference-model', 'sliced-model')

    grouped_data = {}

    for model_type in os.listdir(root_dir):

        if model_type in dir_filter:
            type_dir = os.path.join(root_dir, model_type)

            type_data = {}

            for architecture in os.listdir(type_dir):
                architecture_dir = os.path.join(type_dir, architecture)

                architecture_data = {}

                for lr in learning_rates:
                    selector = get_lr_filter(lr)
                    lr_data = read_files(architecture_dir, selected=selector)

                    if aggregate is not None:
                        lr_data = aggregate_df_dict(lr_data, aggregate=aggregate)

                    architecture_data[lr] = lr_data

                type_data[architecture] = architecture_data

            grouped_data[model_type] = type_data

    return grouped_data['sliced-model'], grouped_data['reference-model']


# Takes dict of dicts of dataframes and returns dict of dataframes containing
# the data of the best learning rate for each architecture
def take_best_lrs(dict_of_dict_of_dfs, accuracy_column='val_acc', peak_acc_count=5):
    best_lrs = {}

    for architecture, architecture_data in dict_of_dict_of_dfs.items():
        best_peak_acc = -1

        for lr, lr_data in architecture_data.items():
            peak_acc = calc_peak_acc(lr_data, peak_acc_count, accuracy_column='val_acc')

            if peak_acc > best_peak_acc:
                best_lr = lr
                best_peak_acc = peak_acc

        # assign best lr of architecture as data for architecture
        best_lrs[f"{architecture} lr: {best_lr}"] = architecture_data[best_lr]
    return best_lrs


def calc_peak_acc(df, peak_acc_count, accuracy_column):
    data = df[accuracy_column].to_numpy()
    data = np.sort(data)
    data = data[::-1]
    return data[peak_acc_count-1]


def convert_dict_to_labeled_xy_list(dict):
    xs = []
    ys = []
    keys = []

    for key, (x, y) in dict.items():
        xs.append(x)
        ys.append(y)
        keys.append(key)

    return xs, ys, keys



def plot_2d_comparison(aggregate='median', time_column='time', accuracy_column='val_acc', peak_count=5, savedir=None, filter=None, filter_str='', point_legend=False, naming=lambda x: x):
    sliced_data, reference_data = read_all_data(aggregate=aggregate)
    sliced_data = take_best_lrs(sliced_data, accuracy_column=accuracy_column, peak_acc_count=peak_count)
    reference_data = take_best_lrs(reference_data, accuracy_column=accuracy_column, peak_acc_count=peak_count)

    sliced_data = {key: (df[time_column].median(), 100 * calc_peak_acc(df, peak_count, accuracy_column)) for (key, df) in sliced_data.items()}
    reference_data = {key: (df[time_column].median(), 100 * calc_peak_acc(df, peak_count, accuracy_column)) for (key, df) in reference_data.items()}

    # filter data
    if filter is not None:
        sliced_data = {key: value for key, value in sliced_data.items() if filter(value)}
        reference_data = {key: value for key, value in reference_data.items() if filter(value)}

    sliced_x, sliced_y, sliced_labels = convert_dict_to_labeled_xy_list(sliced_data)
    sliced_color = 'co'

    reference_x, reference_y, reference_labels = convert_dict_to_labeled_xy_list(reference_data)
    reference_color = 'mo'

    aggregate_name = aggregate if type(aggregate) is str else "max"

    accuracy_column = naming(accuracy_column)
    time_column = naming(time_column)
    aggregate_name = naming(aggregate_name)
    filter_str = naming(filter_str)

    title_string = f"{accuracy_column} / {time_column}, aggregated using: {aggregate_name}"
    filename = f"{aggregate_name}-{accuracy_column}-{time_column}-{filter_str}.png"

    plt.plot(sliced_x, sliced_y, sliced_color, label='sliced-model')
    plt.plot(reference_x, reference_y, reference_color, label='reference-model')

    # part to create a number for every point and store in .txt-file
    if point_legend:
        legend_x_offset = -0.04
        legend_y_offset = -0.04

        pt_legend_list = []
        for index, point_label in enumerate(sliced_labels):
            point_number = index + 1
            plt.annotate(point_number, (sliced_x[index] + legend_x_offset, sliced_y[index] + legend_y_offset), size=7)
            pt_legend_list.append(f"{point_number}: SlicedLSTM - {sliced_labels[index]}\n")

        offset = len(sliced_x)
        for index, point_label in enumerate(reference_labels):
            point_number = index + offset + 1
            plt.annotate(point_number, (reference_x[index] + legend_x_offset, reference_y[index] + legend_y_offset), size=7)
            pt_legend_list.append(f"{point_number}: Standard LSTM - {reference_labels[index]}\n")

        point_legend_filename = f"{aggregate_name}-{accuracy_column}-{time_column}-{filter_str}-point_legend.txt"
        # write pt_legend file
        if savedir is not None:
            pt_legend_file = os.path.join(savedir, point_legend_filename)
            with open(pt_legend_file, 'w') as file:
                file.writelines(pt_legend_list)

    plt.legend()
    plt.xlabel(f"epoch {time_column} [s]")
    # plt.ylabel(f"{accuracy_column} [%]")
    plt.ylabel(f"peak validation accuracy [%]")
    plt.title(title_string)

    if savedir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savedir, filename))

    plt.clf()

def plot_runtime_valacc_comparison(savedir=None, point_legend=False, naming=None):
    aggregates = ['mean', 'median', get_max_valacc_run]
    times = ['time', 'fwd_time', 'bwd_time']

    for aggregate in aggregates:
        for time in times:
            plot_2d_comparison(aggregate=aggregate, time_column=time, savedir=savedir, point_legend=point_legend,
                               filter=lambda val: val[0] < 14, filter_str='max14s', naming=naming)


def get_max_valacc_run(dfs: dict, peak_acc_count=5):
    run_list = [(key, calc_peak_acc(df, peak_acc_count=peak_acc_count, accuracy_column='val_acc')) for (key, df) in dfs.items()]
    sort_key = lambda i: i[1]
    run_list.sort(key=sort_key, reverse=True)
    best_df = run_list[0][0]

    return dfs[best_df]



if __name__ == '__main__':
    # dfs = read_files("results")
    # line_plot(dfs, column="val_acc", show=True, legend=True, xlabel="epoch", ylabel="validation accuracy")

    # plot_lr_comparison("results/sliced-model", aggregate="median", show=False, save=True)

    # plot_run_comparison("results/sliced-model", aggregate="mean", show=False, save=True)

    # plot_lr_comparison("results/sliced-model", aggregate="median", show=False, save=True)

    name_dict = {'fwd_time': 'fwd path time', 'bwd_time': 'bwd path time', 'time': 'total time',
                 'val_acc': 'peak val acc', 'max': 'peak val acc'}

    fancy_naming = lambda n: name_dict[n] if n in name_dict else n

    plot_runtime_valacc_comparison(point_legend=True, naming=fancy_naming, savedir="graphs/2d comparisons")
