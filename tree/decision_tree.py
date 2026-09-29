import numpy as np
from numpy.typing import NDArray , ArrayLike

class Node :
    """Creating Nodes for the Decision Tree Classifier"""
    def __init__(self):
        # Decision Node
        self.feature = None # Index of the feature to split on
        self.threshold = None # THreshold where the node splits to other nodes
        self.right = None # Point to the Node in the right
        self.left = None # Point to the Node in the left
    
        # Leaf Node
        self.is_leaf = False # Is a leaf node (End Node)
        self.value = None # Value of Node - Only if it is a leaf node

class DecisionTreeClassifier:
    def __init__(self):
        self._fitted = False

    def fit(self , X:ArrayLike=None , y:ArrayLike=None):
        self.X = X
        self.y = y

        x_rows , x_cols = np.shape(self.X)
        y_rows , y_cols = np.shape(self.y)
            
        

        self._fitted = True


x = np.random.normal(size=(100 , 2))
y = np.random.normal(size=(100 , 1))

model = DecisionTreeClassifier()

model.fit(x , y)

