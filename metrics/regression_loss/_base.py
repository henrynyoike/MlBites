import numpy as np
from numpy.typing import ArrayLike

def mean_squared_error(self , y_true:ArrayLike=None, y_pred:ArrayLike=None):
    """Calculate the Mean Squared Error"""
    return np.mean((y_pred -y_true)**2)

def mean_absolute_error(self , y_true:ArrayLike=None, y_pred:ArrayLike=None):
    """Calculate the Mean Absolute Error""" 
    return np.mean(np.abs(y_pred -y_true))

def r2_score(self , y_true:ArrayLike=None, y_pred:ArrayLike=None):
    """Calculate the R2 Score"""
    return None



