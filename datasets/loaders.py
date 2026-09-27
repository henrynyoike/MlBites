import pandas as pd
import os

class load_iris :
    def __init__(self , return_X_y:bool=False):
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(current_dir , "data/iris.csv")

        self.full_data = pd.read_csv(data_dir)
    
        self.data = self.full_data.iloc[: , :-1].to_numpy()
        self.target = self.full_data.iloc[: , -1].to_numpy()
        
        self.feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        self.target_names = ['setosa' , 'versicolor' , 'virginica']
        
        if return_X_y :
            return (self.data , self.target)

        else :
            return None

