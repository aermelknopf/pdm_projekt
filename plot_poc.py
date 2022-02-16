import matplotlib.pyplot as plt
import os
import pandas

def read_files(dir, selected=None, round_decimals=None):
    dfs = {}

    for filename in os.listdir(dir):

        if selected is not None:
            if filename in selected:
                df = pandas.read_csv(f"{dir}/{filename}", sep=" ")
                if round_decimals is not None:
                    df = df.round(decimals=round_decimals)

                dfs[filename] = df

    return dfs


def line_plot(df_dict, column, value_factor=1, xlabel=None, ylabel=None, title=None, legend=False, show=False, savepath=None):
    plot_data = []

    for label, df in df_dict.items():
        xs = df['epoch'].values
        ys = df[column].values * value_factor

        if type(legend) is dict:
            if label in legend:
                label = legend[label]

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

    # needs to happen before plt.show() else only white image will be plotted
    if savepath is not None:
        plt.savefig(savepath)

    if show:
        plt.show()




if __name__ == "__main__":
    result_dir = "results"

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
