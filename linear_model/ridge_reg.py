import numpy as np
from numpy.typing import NDArray , ArrayLike
from pandas import DataFrame
from sklearn import linear_model

class RidgeRegression : 
    def __init__(self):
        self._coeff = None
        self.bias = None

    def fit(self , X:ArrayLike|DataFrame=None , 
            y:ArrayLike|DataFrame=None ,
            alpha:float = 0.001,
            iterations:int=100):

        self.X = X
        self.y = y

        self.alpha = alpha
        self.iterations = iterations

        # Get the shapes of X and Y
        x_rows , x_cols = np.shape(self.X)
        y_rows , y_cols = np.shape(self.y)

        # Initializa the coefficients and bias
        self._coeff = np.linalg.inv(self.X.T@self.X) @ (self.X.T@self.y)
        self.bias = 0 

        for _ in range(self.iterations):
            y_pred = np.dot(self.X , self._coeff) + self.bias

            error = y_pred - y

            dw = (2 / x_rows) * (self.X.T @ error)
            db = (2 / x_rows) * np.sum(error)

            # Gradient Descent
            self._coeff -= (dw * self.alpha)
            self.bias -= (db * self.alpha)

        return self._coeff , self.bias
    
    def predict(self , X:NDArray|ArrayLike):
        return np.dot(X , self._coeff) + self.bias

x = np.random.uniform(size=(1000 , 5))
y = np.random.uniform(size=(1000 , 1))

model = RidgeRegression()
model2 = linear_model.Lars()

model.fit(x , y)
model2.fit(x ,y)

y_pred = model.predict(x)
y_pred2 = model2.predict(x)

print("Y : " , np.mean((y_pred - y) ** 2))
print("Y2 :" , np.mean((y_pred2 - y) ** 2))


