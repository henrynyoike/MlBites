import numpy as np
from numpy.typing import ArrayLike
from sklearn import metrics

def mean_squared_error(y_true:ArrayLike=None, y_pred:ArrayLike=None):
    """Calculate the Mean Squared Error"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return np.mean((y_true-y_pred)**2)

def mean_absolute_error(y_true:ArrayLike=None, y_pred:ArrayLike=None):
    """Calculate the Mean Absolute Error""" 
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true:ArrayLike=None, y_pred:ArrayLike=None):
    """Calculate the R2 Score"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot)

def root_mean_squared_error(y_true:ArrayLike=None, y_pred:ArrayLike=None):
    """Calculate the mean of the mean squared error"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return np.sqrt(mean_squared_error(y_true , y_pred))

def median_absolute_error(y_true:ArrayLike=None, y_pred:ArrayLike=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    return np.median(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Replicate scikit-learn's zero-handling behavior
    # Replace 0 in y_true with machine epsilon to avoid division by zero
    eps = np.finfo(np.float64).eps
    y_true_safe = np.where(y_true == 0, eps, y_true)
    
    # Calculate the mean absolute percentage error as a fraction
    return np.mean(np.abs((y_true - y_pred) / y_true_safe))

def mean_squared_log_error(y_true:ArrayLike, y_pred:ArrayLike):
    """Calculate the Mean Squared Log Error"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def adjusted_r2_score(y_true:ArrayLike=None, y_pred:ArrayLike=None , features:int=1):
    """Calculate the adjusted_r2_score"""
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred cannot be None")

    r2 = r2_score(y_true , y_pred)
    n = np.shape(y_true)[0]
    adjusted = 1 - ((1 - r2) * (n - 1)) / (n - features - 1)
    return adjusted

