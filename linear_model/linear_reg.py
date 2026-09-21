import numpy as np
from numpy.typing import NDArray , ArrayLike
import sys
from pandas import DataFrame
from sklearn import linear_model

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
        #self._coeff = np.zeros((self.x_shape[1] , 1))
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

x = np.random.normal(size=(5000 , 10))
y = np.random.normal(size=(5000 , 2))

model = LinearRegression()
lr = linear_model.LinearRegression()

model.fit(x , y)
lr.fit(x ,y)

y_pred = model.predict(x)
y2 = lr.predict(x)

mse = np.mean((y_pred - y) **2)
mse2 = np.mean((y2 - y) **2)

print(mse)
print(mse2)
