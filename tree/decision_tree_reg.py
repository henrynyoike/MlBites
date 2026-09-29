import numpy as np
from numpy.typing import NDArray , ArrayLike

class Node :
    """Creating Nodes for the Decision Tree Regressor"""
    def __init__(self , feature=None , threshold=None , right=None , left=None , is_leaf = False , value = None):
        self.feature = feature # Index of the feature to split on
        self.threshold = threshold # THreshold where the node splits to other nodes
        self.right = right # Point to the Node in the right
        self.left = left # Point to the Node in the left
        self.is_leaf = is_leaf # Is a leaf node (End Node)
        self.value = value # Value of Node - Only if it is a leaf node

class DecisionTreeRegressor:
    def __init__(self , min_samples_split:int=2 , max_depth:int=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self._fitted = False

    def fit(self , X:ArrayLike=None , y:ArrayLike=None):
        self.X = np.array(X)
        self.y = np.array(y)
        
        self.root_node = self._build_tree(X=self.X , y=self.y , current_depth=0)

        self._fitted = True
    
    def _build_tree(self ,X:ArrayLike=None , y:ArrayLike=None , current_depth:int=0):
        if current_depth >= self.max_depth or X.shape[0] < self.min_samples_split :
            return Node(is_leaf=True,value=np.mean(y)) #
        
        best_feature , best_threshold = self._best_split(X , y)

        if best_feature == None or best_threshold == None :
            return Node(is_leaf=True ,value=np.mean(y))
        
        left_data = X[: , best_feature] <= best_threshold # Get the boolean values that agree with the threshold
        right_data = X[: , best_feature] > best_threshold

        left = self._build_tree(X[left_data] , y[left_data] , current_depth+1) # Build the left node then the right one
        right = self._build_tree(X[right_data] , y[right_data] , current_depth+1)

        return Node(feature=best_feature ,threshold=best_threshold ,right=right ,left=left)

    def _best_split(self ,X:ArrayLike=None,y:ArrayLike=None):
        best_feature = None
        best_threshold = None
        best_score = float("inf") # Do not set to zero to allow setting of the best_feature and threshold

        n_samples, n_features = np.shape(X)

        for feature in range(n_features):
            for threshold in X[: ,feature]:
                # Split dataset based on the threshold value
                left = y[X[: , feature] <= threshold] 
                right = y[X[: , feature] > threshold]
                
                if len(left) == 0 or len(right)== 0:
                    continue # Skip if there is no useful split

                score = (len(left) * np.var(left) + len(right) * np.var(right)) / n_samples

                if score < best_score : # If thew score is greater than the previous one
                    best_score = score
                    best_feature , best_threshold = feature , threshold
        return best_feature , best_threshold

    def predict_one(self ,node=None ,X:ArrayLike=None):
        if node.value is not None :
            return node.value

        #print(X[node.feature] <= node.threshold)
        if X[node.feature] <= node.threshold :
            return self.predict_one(node.left , X)
       #print(X[node.feature] <= node.threshold)  

        return self.predict_one(node.right , X)

    def predict(self , X:ArrayLike=None):
        #return np.array([y for y in self.predict_one(self.root_node , np.array(X))])
        return np.array([self.predict_one(self.root_node , x) for x in np.array(x)])

x = np.random.normal(size=(100 , 2))
y = np.random.normal(size=(100 , 1))

model = DecisionTreeRegressor()

model.fit(x , y)

y_pred = model.predict(x)

print(y_pred)


