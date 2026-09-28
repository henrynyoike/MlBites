import numpy as np 
from numpy.typing import ArrayLike

def make_regression(n_samples:int=100 , 
            n_features:int=100 , 
            n_targets=1 , 
            shuffle:bool = True):

    if n_features > n_samples : 
        raise ValueError("Number of Features is greater that the total number of observations in the dataset")
        sys.exit(0)   
   
    coeff = np.random.randn(n_features , 1)
    bias = np.random.randn()

    # Get the random values of the datasets
    X = np.random.randn(n_samples , n_features)
    y = np.dot(X , coeff ) + bias

    data = (X , y)

    return data

