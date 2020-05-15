"""Core module for own metrics implementation"""
from sklearn.metrics import mean_squared_error

import numpy as np

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))