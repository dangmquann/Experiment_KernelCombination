import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path, input=True):
    if input:
        data = pd.read_csv(path, header=None)
        data = data.values
        return data
    else:
        data = pd.read_csv(path)
        data = data['encoded']
        le = LabelEncoder()
        y = le.fit_transform(data)
        return y

def sort_XyZ(X_list,y,Z=None):
    if Z is None:
        Z = np.array(range(len(y)))
    indices = np.argsort(y)
    for i in range(len(X_list)):
        X_list[i] = X_list[i][indices,:]
    y = y[indices]
    Z = Z[indices]
    return X_list,y,Z

