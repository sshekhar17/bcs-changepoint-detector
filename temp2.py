import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle 


df = pd.read_csv('./data/data_banknote_authentication.txt', sep=',',skiprows=2,
                    header=None)

data = df.to_numpy()
labels, features = data[:, -1], data[:, :-1] 
# features = np.delete(features, 1, 1)
features = features.astype(float)
f0, f1 = features[labels==0], features[labels==1]
m0, s0 = f0.mean(axis=0, keepdims=True), f0.std(axis=0, keepdims=True)
m1, s1 = f1.mean(axis=0, keepdims=True), f1.std(axis=0, keepdims=True)
f0, f1 = (f0-m0)/s0, (f1-m1)/s1

n0, n1 = len(f0), len(f1) 
# assert (2*N<=n0) and (N<=n1)
# create the dataset 
f0_, f1_ = shuffle(f0), shuffle(f1)
