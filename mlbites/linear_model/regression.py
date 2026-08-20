import numpy as np
from numpy.typing import ArrayLike , NDArray
#from sklearn.linear_model import LinearRegression

class LinearRegression:
    """
    Linear Regression Model using the Normal Equation and Gradient Descent.
    Error used is the Mean squared error

    ### Example Usage : 

    lr = LinearRegression()
    
    X = np.random.randn(10 , 2)
    y = np.random.randn(10 , 1)

    # Fit the model
    lr.fit(X , y)

    # Predict Some Output
    new_data = np.random.randn(10 , 2)
    y_pred = lr.predict(new_data)

    """

    def __init__(self , 
                 learning_rate:float = 0.001,
                 iterations:int=100 ,
                 copy_X:bool=True):
        """
        learning_rate - Steps to use in Gradient Descent
        iterations - Number of iterations used in the training loop
        copy_X - Copy the training data during training

        """
        self.fitted = False 
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coeff = None
        self.bias = None
        self.copy_X = copy_X
    
    def _compute_gradient(self , X:ArrayLike=None , errors:ArrayLike=None):
        """
        Coefficient Gradient = 2/m * X.T @ errors
        Bias Gradient = mean_squared_error / m

        errors = y_pred - y_true
        m = total number of observations

        """
        # Get the Number of observations
        m = len(X)

        # Mean squared error for the gradient
        mean_squared_error = np.mean(errors ** 2)

        w_grad = (2/m) * X.T @ errors
        b_grad = mean_squared_error / m

        return w_grad , b_grad 

    def _gradient_descent(self , w_grad , b_grad):
        """Gradient Descent of the model's coefficients and bias."""
        self.coeff -= w_grad * self.learning_rate
        self.bias -= b_grad * self.learning_rate  
        
        return self.coeff , self.bias

    def fit(self , X:ArrayLike=None , y:ArrayLike=None) -> NDArray[np.int64]:
        # Get the coefficients through the Normal Equation
        # (X^T X)^-1(X^T Y)
        if self.copy_X :
            self.X_train = X.copy()
            self.y_train = y.copy()
        else :
            self.X_train = X
            self.y_train = y

        self.X_shape = np.shape(X)
        self.y_shape = np.shape(y)
        
        # Use the Normal Equation to get the Coefficient
        self.coeff = np.linalg.inv((X.T@X)) @ (X.T@y)

        # Create an array of zeros to represent the bias
        self.bias = np.zeros((1 , self.y_shape[1])) #With a shape (1 , columns_of_y)

        for i in range(self.iterations):
            self.error = self._predict(self.X_train) - self.y_train
            w_grad , b_grad= self._compute_gradient(X = self.X_train , errors=self.error)
            self.coeff , self.bias = self._gradient_descent(w_grad = w_grad , b_grad = b_grad)

        self.fitted = True
        
        return self.coeff

    def _predict(self , X:ArrayLike=None) -> NDArray[np.float64]:
        return X@self.coeff + self.bias
    
    def predict(self ,X:ArrayLike=None) -> NDArray[np.float64]:
        """Predict an Output 'Y' from the Input 'X' using the trained model"""
        if not self.fitted :
            print("Please Train the Model First in order to predict an output")
            return False
        return self._predict(X)


x = np.random.randn(10 , 2)
y = np.random.randn(10 , 1)

model = LinearRegression()

model.fit(x , y)

y_pred = model.predict(x)

mean_squared_error = np.mean((y_pred-y) ** 2)

print("Mean Squared Error : " , mean_squared_error)


