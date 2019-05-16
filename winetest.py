# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:46:52 2019

@author: saurav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset1=pd.read_csv('winequality-white.csv')
dataset1=dataset1.iloc[:,0].values
dataset1=dataset1.reshape(4898,1)

dataset2=pd.read_csv('winequality-red.csv')
dataset2=dataset2.iloc[:,0].values
dataset2=dataset2.reshape(1599,1)

X=[]

for i in range(0,4898):
    t=dataset1[i][0].split(';')
    X.append(t)
    
for i in range(0,1599):
    t=dataset2[i][0].split(';')
    X.append(t)

X=np.asarray(X)
X=X.astype(np.float)

y=X[:,11]
X=X[:,0:11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)