import numpy as np
from numpy.typing import ArrayLike , NDArray


class LogisticRegression :
    def __init__(self):
        self.fitted = False
        self.learning_rate = 0.001
    
    def _sigmoid(self , z:ArrayLike=None):
        return 1 / (1 + np.exp(-z))
    
    def _log_loss(self , y_pred:ArrayLike=None , y_true:ArrayLike=None):
        """
        Formulae = -mean[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        """
        return -np.mean((y_true * np.log(y_pred)) + 
                        (1 - y_true) * 
                        (np.log(1 - y_pred)))

    def _compute_gradient(self , X:ArrayLike=None , loss:ArrayLike=None):
        """
        Coefficient Gradient = 2/m * X.T @ loss
        Bias Gradient = loss / m

        errors = log_loss
        m = total number of observations

        """
        # Get the Number of observations
        m = np.shape(X)[0]

        #w_grad = np.dot(X.T,loss) /  m
        #b_grad = loss / m
        w_grad =  (1/m) * np.dot(X.T , loss)
        b_grad = (1/m) * np.sum(loss)


        return w_grad , b_grad 
    
    def fit(self , X:ArrayLike,y:ArrayLike ,iterations=1000):
        self.y_shape = np.shape(y)

        #self.w = np.linalg.inv((X.T @ X)) @ np.dot(X.T , y)
        self.w = np.zeros(X.shape[1])
        self.bias = np.zeros((1 , self.y_shape[0]))
        
        for i in range(iterations):
            Z = np.dot(X , self.w) + self.bias
            
            y_pred = self._sigmoid(Z)
            entropy_loss = self._log_loss(y_pred=y_pred,y_true=y)

            dw , db = self._compute_gradient(X , loss = (y_pred - y))
            print("W grad shape : " , np.shape(dw))
            self.w -= self.learning_rate * dw
            self.bias = self.learning_rate *db

        self.fitted = True

x = np.random.randn(5 , 2)
y = np.array([[0],[0],[1],[1],[0]])

model = LogisticRegression()

model.fit(x , y)





            








