import numpy as np

def mean_absolute_error(y_pred, y_test):
    return np.mean(np.abs(y_pred - y_test))

def mean_squared_error(y_pred, y_test):
    return np.mean((y_pred - y_test) ** 2)
