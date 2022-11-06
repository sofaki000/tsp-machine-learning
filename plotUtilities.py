from IPython.core.display import Image
from time import gmtime, strftime
from keras.utils import plot_model
import matplotlib.pyplot as plt

def plot_model_with_me(model, file_name):
    print(model.summary())
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=file_name)
    Image(retina=True, filename=file_name)

def save_plot_with_y_axis(y_axis, title, file_name,xLabel, yLabel):
    plt.plot(y_axis)
    plt.title(title)
    file_name = strftime(f"{file_name}%Y-%m-%d %H:%M:%S", gmtime())
    file_name = file_name.replace("-", "_")
    file_name = file_name.replace(":", "_")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(f'{file_name}.png')