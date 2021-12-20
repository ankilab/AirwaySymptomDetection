import numpy as np
import os


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def get_labels_array(array, string):
    if string == "Cough":
        array[0] = 1
    elif string == "Dry swallow":
        array[1] = 1
    elif string == "Throat clear":
        array[2] = 1
    elif string == "No event":
        array[3] = 1
    elif string == "Speech":
        array[3] = 1
    elif string == "Silence":
        array[4] = 1
    return array
