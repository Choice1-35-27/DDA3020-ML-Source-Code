import numpy as np

def get_MSE(y_true, y_pred, accuracy_digits=2):
    """
    calculate mean square error
    """
    return np.mean((y_true - y_pred) ** 2).round(accuracy_digits)