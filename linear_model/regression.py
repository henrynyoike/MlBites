import numpy as np
from numpy.typing import ArrayLike , NDArray
import pandas as pd
from pandas import DataFrame


class LinearRegression : 
    def __init__(self):
        self._fitted = False
        self._coeff = None
        self.bias = 0
        self.lr = 0.001
        self.iterations = 100

    def _validate_data(self,
            X:ArrayLike|DataFrame=None , 
            y:ArrayLike|DataFrame=None) -> bool: # Check if the dataset is valid for training
        
        x_rows , x_cols = np.shape(X)
        y_rows , y_cols = np.shape(y)
        
        # Length of the Dependent and Independent variables
        if x_rows != y_rows:
            print(f"Error : X and Y must be of same number of rows , X is {self.x_shape[0]} and Y is {self.y_shape[0]}")
            return False

        # Dependent variable must have shape of (n , 1)
        if y_cols != 1 :
            print(f"Error : Dependent Variable 'Y' must be of shape (n , 1) , yours is (n , {y_cols})")
            return False

        return True

    def fit(self , 
            X:ArrayLike|DataFrame=None , 
            y:ArrayLike|DataFrame=None, 
            learning_rate:float = 0.001,
            iterations:int = 100):
        
        self.lr = learning_rate
        self.iterations = iterations

        self.X = X
        self.y = y

        self.x_shape = np.shape(X)
        self.y_shape = np.shape(y)

        if not self._validate_data(self.X , self.y):
            sys.exit(0)

        # Initiate the coefficients and bias to zero
        self._coeff = np.linalg.inv((self.X.T@self.X)) @ (self.X.T@self.y) 
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X , self._coeff) + self.bias
            
            # Get the gradients of the coefficients and bias
            n_samples = self.x_shape[0]
            
            dw = (2 / n_samples) * np.dot(X.T , (y_pred - self.y))
            db = (2 / n_samples) * np.sum(y_pred - self.y)
            
            # Gradient descent of the coefficients and bias
            self._coeff -= (dw * self.lr)
            self.bias -= (db * self.lr)
        
        self._fitted = True
        return self._coeff , self.bias

    def predict(self , X:ArrayLike|DataFrame = None):
        if not self._fitted :
            print("Please Train model first with 'LinearRegression.fit(X , y)' in order to predict")
            sys.exit0()

        return np.dot(X , self._coeff) + self.bias

# Creating a Logistic regression model
class LogisticRegression :
    def __init__(self):
        self._fitted = False
    
    def sigmoid(self , Z:NDArray=None):
        return  1 / (1 + np.exp(-Z))
    
    def fit(self , X:ArrayLike|DataFrame = None , 
            y:ArrayLike|DataFrame = None , 
            iterations:int = 100 , 
            learning_rate:float = 0.001):
        
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.iterations = iterations

        x_shape = np.shape(self.X)
        y_shape = np.shape(self.y)

        n_samples = x_shape[0]

        self._coeff = np.linalg.inv(self.X.T@self.X) @ (self.X.T@y)
        self.bias = 0

        for _ in range(self.iterations):
            z = np.dot(self.X , self._coeff) + self.bias
            y_pred = self.sigmoid(z)

            # Get the gradients of the weights and bias
            dw = (1 / n_samples ) * np.dot(X.T , (y_pred - y))
            db = (1 / n_samples ) * np.sum(y_pred -y)

            # Gradient Descent
            self._coeff -= (dw * self.learning_rate)
            self.bias -= (db * self.learning_rate)

        self._fitted = True

    def predict_proba(self , X:ArrayLike|DataFrame=None):
        # Predict the raw probabilities
        out = np.dot(X , self._coeff) + self.bias
        sigmoid_fn = self.sigmoid(out)

        return sigmoid_fn
    
    def _predict(self , X:ArrayLike|DataFrame=None):
        out = np.dot(X , self._coeff) + self.bias
        sigmoid_fn = self.sigmoid(out)
        return (sigmoid_fn >= 0.5).astype(int)

    def predict(self , X:ArrayLike|DataFrame=None):
        return np.array(self._predict(X)) 

