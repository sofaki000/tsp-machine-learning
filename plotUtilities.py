from IPython.core.display import Image
from tensorflow import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt

def plot_model_with_me(model, file_name):
    print(model.summary())
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=file_name)
    Image(retina=True, filename=file_name)