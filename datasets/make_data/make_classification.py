import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def make_classification(
    n_samples:int = 100, 
    n_features:int = 20,
    n_classes:int = 2, 
    n_repeated:int = 0 , 
    shuffle:bool = True
):
    if n_repeated > n_samples : 
        raise ValueError("Number of Repeated observations is greater that the total number of observations in the dataset")
        sys.exit(0)   
   
    out_shape = (n_samples-n_repeated , n_features) # Samples to be less by n_repeated to make room for repeated observations
    
    # Get the random values of the datasets
    X = np.random.uniform(size=out_shape) # Uniform distribution of samples
    y = np.random.randint(low = 0 , high = n_classes , size=(n_samples , 1) , dtype=int)

    if n_repeated > 0 :
        repeat_set = np.random.uniform(size=(1 , n_features)) # Get one sample to repeat
        repeated = np.repeat(repeat_set , n_repeated , axis=0)
        X = np.append(X , repeated , axis=0) # Append the repeated sample to the main dataset

    if shuffle: 
        np.random.shuffle(X)
    
    data = (X , y)
    return data

x ,y = make_classification(n_samples=1000000 , n_features=10 , n_classes=3 , n_repeated=0)

model = DecisionTreeClassifier()

model.fit(x , y)

y_pred = model.predict(x)

accuracy = accuracy_score(y_pred , y)

print("Accuracy : " , accuracy)

