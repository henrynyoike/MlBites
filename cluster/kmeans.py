from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X ,y = make_blobs(n_samples=100 , n_features=1 , centers=10)

x_rows , x_cols = np.shape(X)

K = 10 # Number of clusters
max_iters = 10 # Max number of iterations

centroids = np.random.randn(K , x_cols) # Set the shape of the centroids

for i in range(K):
    centroid = X[np.random.choice(range(x_rows))]
    centroids[i] = centroid

def euclidean_distance(point , centroid):
    return np.sqrt((point - centroid)**2)

def get_nearest_centroid(value , centroids):
    nearest_centroid = 0
    
    for index , centroid in enumerate(centroids) :
        euc_dist = euclidean_distance(value , centroid)
        if euclidean_distance(value , nearest_centroid) > euc_dist:
            nearest_centroid = index
        previous_centroid = euc_dist
    
    return nearest_centroid

def get_centroids(values):
    return np.mean(values)

def create_clusters(centroids , X):
    clusters = [[] for i in range(K)]

    for index , value in enumerate(X):
        closest_centroid = get_nearest_centroid(value , centroids)

        clusters[closest_centroid].append(value)
        
        if value == None :
            print("Gotcha : " , index)
            print("!" ** 3000)
            quit()
    return clusters

for _ in range(max_iters):
    # Create Clusters
    clusters = create_clusters(centroids , X)
    print(clusters) 
    # Copy previous centroids
    previous_centroids = centroids
    # Create new centroids based on the clusters
    centroids = [get_centroids(i) for i in clusters]
    


