import numpy as np
from numpy.typing import ArrayLike , NDArray
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn import linear_model

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

        self.W = np.zeros((x_shape[1] , y_shape[1]))
        self.bias = 0

        for _ in range(self.iterations):
            z = np.dot(self.X , self.W) + self.bias
            y_pred = self.sigmoid(z)

            # Get the gradients of the weights and bias
            dw = (1 / n_samples ) * np.dot(X.T , (y_pred - y))
            db = (1 / n_samples ) * np.sum(y_pred -y)

            # Gradient Descent
            self.W -= (dw * self.learning_rate)
            self.bias -= (db * self.learning_rate)

        self._fitted = True

    def predict_proba(self , X:ArrayLike|DataFrame=None):
        # Predict the raw probabilities
        out = np.dot(X , self.W) + self.bias
        sigmoid_fn = self.sigmoid(out)

        return sigmoid_fn
    
    def _predict(self , X:ArrayLike|DataFrame=None):
        out = np.dot(X , self.W) + self.bias
        sigmoid_fn = self.sigmoid(out)
        return (sigmoid_fn >= 0.5).astype(int)

    def predict(self , X:ArrayLike|DataFrame=None):
        return np.array(self._predict(X)) 

#x = np.random.rand(10000 , 1000)
#y = np.random.randint(0 , 2 , (10000 , 1))
#
#model = LogisticRegression()
#model2 = linear_model.LogisticRegression()
#
#model.fit(x , y)
#model2.fit(x , y.ravel())
#
#y_pred1 = model.predict(x)
#y_pred2 = model2.predict(x)
#
#score1 = accuracy_score(y , y_pred1)
#score2 = accuracy_score(y , y_pred2)
#
#print(score1)
#print(score2)

