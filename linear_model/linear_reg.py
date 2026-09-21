import numpy as np
from numpy.typing import NDArray , ArrayLike
from pandas import DataFrame

class LinearRegression : 
    def __init__(self):
        self._fitted = False
        self.coeff = None
        self.bias = 0

    def fit(self , 
            X:ArrayLike|DataFrame=None , 
            y:ArrayLike|DataFrame=None, 
            learning_rate:float = 0.01,
            iterations:int = 100):
        
        self.lr = learning_rate
        self.iterations = iterations

        self.X = X
        self.y = y

        self.x_shape = np.shape(X)
        self.y_shape = np.shape(y)

        # Initiate the coefficients and bias to zero
        self._coeff = np.zeros((self.x_shape[1] , 1))
        self.bias = 0#np.zeros((1 , self.y_shape[1]))

        for _ in range(self.iterations):
            y_pred = np.dot(X , self._coeff) + self.bias

            # Get the gradients of the coefficients and bias
            n_samples = self.x_shape[0]
            
            dw = (1 / n_samples) * np.dot(X.T , (y_pred - self.y))
            db = (1 / n_samples) * np.sum(y_pred - self.y)
            
            # Gradient descent of the coefficients and bias
            self._coeff -= (dw * self.lr)
            self.bias -= (db * self.lr)
        
        self._fitted = True
        return self._coeff , self.bias

    def predict(self , X:ArrayLike|DataFrame = None):
        return np.dot(X , self._coeff) + self.bias
        
#x = np.random.randint(0 , 10 , (10 , 5))
#y = np.random.randint(0 , 10 , (10 , 1))
x = np.random.randn(10 , 5)
y = np.random.randn(10 , 1)

model = LinearRegression()

model.fit(x , y)

y_pred = model.predict(x)

print(y)
print(y_pred)


