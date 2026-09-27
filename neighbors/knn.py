import numpy as np
import sys
from collections import Counter
from numpy.typing import NDArray , ArrayLike
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class KNeighborsClassifier :
    def __init__(self , n_neighbors:int = 5):
        self.n_neighbors = n_neighbors
        self._fitted = False
        self.p = 2

    def minkowski_distance(self , point_a : ArrayLike=None, point_b:ArrayLike=None , p:int=2):
        return np.sum(np.abs((point_a - point_b) ** p) ** (1 / p))

    def get_distances(self ,point:ArrayLike=None):
        dist_types = [("index" , int) , ("distance" , float)] # Create the dtypes of the index and distance
        distances = np.array([], dtype=dist_types)
        for idx , data in enumerate(self.X):
            dist = self.minkowski_distance(data , point , p = self.p)
            dist_arr = np.array([(idx , dist)] , dtype=dist_types)
            #print("Distance shape : " , np.shape(dist_arr))
            distances = np.concat((distances , dist_arr) , axis=0)
        return distances

    def get_closest_neighbors(self , distances:ArrayLike=None):
        """Get the closest datapoint to the new data given the distances of all the points from the new input """
        arranged_distances = np.sort(distances , axis=0 , order="distance") # Arrange based on the distance columns created in 'get_distances()'
        return arranged_distances[:self.n_neighbors]

    def fit(self , X:ArrayLike=None , y:ArrayLike=None , p:int=2):
        """Just Getting the Data Points , nothing much"""
        self.X = np.array(X)
        self.y = np.array(y)
        self.p = p

        self.x_rows , self.x_cols = np.shape(self.X)
        
        self._fitted = True

    def get_y_neighbors(self , neighbors:ArrayLike=None):
        np_indices = [idx for idx , data in neighbors]
        indices = [int(idx) for idx in np_indices]
        y_neighbors = [int(self.y[idx]) for idx in indices]
        return y_neighbors

    def predict(self , X):
        """Predict an output from the Nearest Neighbor"""
        if not self._fitted :
            print("Please Train model first through 'KNeighborsClassifier().fit(X ,y)' ")
            sys.exit(0)
        
        predictions = []
        for x_point in X:
            distances = self.get_distances(point=x_point)
            nearest_neighbors = self.get_closest_neighbors(distances = distances)
            y_neighbors = self.get_y_neighbors(neighbors=nearest_neighbors)
            common_pred = Counter(y_neighbors).most_common()[0][0]
            predictions.append(common_pred)

        return np.array(predictions)

X ,y = make_classification(n_samples=100, n_features=5)

model = KNeighborsClassifier()

X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size=0.2)

model.fit(X_train ,y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_pred ,y_test)

print(accuracy)
