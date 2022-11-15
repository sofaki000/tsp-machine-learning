from IPython.core.display import Image
from time import gmtime, strftime
from keras.utils import plot_model
import matplotlib.pyplot as plt

def plot_model_with_me(model, file_name):
    print(model.summary())
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=file_name)
    Image(retina=True, filename=file_name)

def save_plot_with_multiple_functions_in_same_figure(results, labels, file_name, title):
    plt.clf()
    # Plotting both the curves simultaneously
    colors = ['r', 'g', 'b']
    n = 10 # plot every 10 points
    for i in range(len(results)):
        plt.plot(results[i][::n],  color=colors[i], label=labels[i])
    # plt.plot(X,  color='g', label='cos')
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Different tours")
    plt.ylabel("Distance travelled")
    plt.title(title)

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    file_name = strftime(f"{file_name}Date%Y_%m_%d_Time%H_%M_%S", gmtime())
    plt.savefig(f'{file_name}.png')

def save_plot_with_y_axis(y_axis, title, file_name,xLabel, yLabel):
    plt.plot(y_axis)
    plt.title(title)

    file_name = strftime(f"{file_name}_Date:%Y-%m-%d_Time:%H:%M:%S", gmtime())
    file_name = file_name.replace("-", "_")
    file_name = file_name.replace(":", "_")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.savefig(f'{file_name}.png')

def save_multiple_plots_for_different_experiments(experiments_mean_losses,titles, file_name, xLabel="Epochs", yLabel="Mean loss"):
    file_name = strftime(f"{file_name}Date%Y_%m_%d_Time%H_%M_%S", gmtime())
    figure, axis = plt.subplots(2, 2)
    k = len(experiments_mean_losses) -1
    figure.tight_layout()
    for i in range(2):
        experiment_mean_losses = experiments_mean_losses[i]
        title = titles[i]

        axis[i, 0].plot(experiment_mean_losses)
        axis[i, 0].set_title(title)
        experiment_mean_losses = experiments_mean_losses[k-i]
        title = titles[k-i]
        axis[i, 1].plot(experiment_mean_losses)
        axis[i, 1].set_title(title)

    figure.savefig(f'{file_name}.png')


