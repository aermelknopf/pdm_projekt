import matplotlib.pyplot as plt
import os
import pandas

def read_files(dir, round_decimals=None):
    dfs = {}

    for filename in os.listdir(dir):
        df = pandas.read_csv(f"{dir}/{filename}", sep=" ")
        if round_decimals is not None:
            df = df.round(decimals=round_decimals)

        dfs[filename] = df

    return dfs


def line_plot(df_dict, column, xlabel=None, ylabel=None, title=None, legend=False, show=False, savepath=None):
    plot_data = []

    for label, df in df_dict.items():
        print(label)
        xs = df['epoch'].values
        ys = df[column].values
        plot_data.append((label, xs, ys))

    for (label, xs, ys) in plot_data:
        plt.plot(xs, ys, label=label)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if legend:
        plt.legend()

    if show:
        plt.show()

    if savepath is not None:
        plt.savefig(savepath)


if __name__ == "__main__":
    result_dir = "results"

    dfs = read_files(result_dir, round_decimals=3)
    # dfs["functional_model.txt"]["val_acc"].plot()
    # plt.show()

    line_plot(dfs, column="val_acc", xlabel="Epoch", ylabel="Validation Accuracy",
              legend=True,
              title="Comparison of Keras functional LSTM model and split LSTM model",
              show=True, savepath="graphs/poc_val_acc")
