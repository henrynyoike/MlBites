from sklearn.datasets import make_blobs
import numpy as np
from numpy.typing import ArrayLike , NDArray

class KMeans :
    def __init__(self , n_clusters:int=3 , max_iters:int=300):
        self.n_clusters = n_clusters
        self.X_train = None
        self.max_iters = max_iters
        self.clusters = None
        self.centroids = None
    
    def create_clusters(self , X , centroids):
        clusters = [[] for i in range(self.n_clusters)]
        
        for index , point in X :
            closest_centroid = self._get_closest_centroid(point , centroids)
            clusters[closest_centroid].append(point)
    
        return clusters

    def fit(self , X:ArrayLike=None):
        self.X_train = X
        self.samples , self.features = np.shape(self.X_train)

        self.centroids = np.random.randn(self.n_clusters ,  )
        
        for i in range(self.n_clusters):
            self.centroids[i] = np.mean(self.X_train[np.random.choice(self.samples)])
        
        self.previous_centroids = self.centroids
        for _ in range(self.max_iters):
            self.clusters = self.create_clusters(X , self.centroids)
            
            # Create new centroids using the values in each cluster
            self.centroids = [np.mean(self.clusters[i]) for i in range(self.n_clusters)] 


            diff = [self.centroids[i] - self.previous_centroids[i] for i in range(len(self.centroids))]
            
            if not np.array(diff).any():
                return self.centroids , self.clusters
            
            self.previous_centroids = self.centroids
        return self.centroids , self.clusters

    def _predict(self , X:ArrayLike=None):
        x_new = X
    
        # Predict the new data
        y = []
        for idx , row in enumerate(X):
            nearest_centroid = self._get_closest_centroid(row , self.centroids)
            y.append(nearest_centroid)

        return np.array(y)
                
    def predict(self , X:ArrayLike=None):
        return self._predict(X)

    def _get_closest_centroid(self , point , centroids):
        min_centroid = 0
        min_dist = self.euclidean_distance(point , centroids[min_centroid])
        for index , centroid in enumerate(centroids):
            dist = self.euclidean_distance(point , centroid) # Get the euclidean distance
            if dist < min_dist :
               min_dist = dist
               min_centroid = index

        return min_centroid
        
    def euclidean_distance(self , point_a , point_b):
        return np.sqrt(np.sum(point_a - point_b) ** 2)
        
X , y = make_blobs(n_features=2)

#X = np.random.randint(1 , 100 , (100 , 2))

model = KMeans()

model.fit(X)

y_pred = model.predict(X)
print(y)
print(y_pred)

