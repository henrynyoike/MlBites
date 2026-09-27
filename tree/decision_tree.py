import numpy as np
from numpy.typing import NDArray , ArrayLike

class DecisionTreeRegressor :
    def __init__(self):
        self._fitted = False

    def fit(self , X:ArrayLike=None , y:ArrayLike=None):
        
        self.X = X
        self.y = y

        x_rows , x_cols = np.shape(self.X)
        y_rows , y_cols = np.shape(self.y)


        trees = {}

        



        self._fitted = True


x = np.random.normal(size=(100 , 2))
y = np.random.normal(size=(100 , 1))

model.fit(x , y)

