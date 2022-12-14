import numpy as np
from sklearn.metrics import mean_squared_error


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))