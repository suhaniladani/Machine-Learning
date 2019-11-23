
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg
from __future__ import print_function, division
import pandas as pd
import random
import math


# In[2]:


df = pd.read_csv("perceptronData.txt", delim_whitespace=True, header=None) 
df.head()


# In[3]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indexes = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    
    return train_df, test_df


# In[4]:


random.seed(3)
train_df, test_df = train_test_split(df, 0.20)


# In[5]:


X_train = np.array(train_df.iloc[: , :-1])
y_train = np.array(train_df.iloc[: , -1])

X_test = np.array(test_df.iloc[: , :-1])
y_test = np.array(test_df.iloc[: , -1])


# In[6]:


def project(X):
    return np.dot(X, w) + b

def predict(X):
    X = np.atleast_2d(X)
    return np.sign(project(X))


# In[7]:


T=3
X = X_train
y = y_train
n_samples, n_features = X.shape
w = np.zeros(n_features, dtype=np.float64)
b = 0.0

for t in range(T):
    for i in range(n_samples):
        if predict(X[i])[0] != y[i]:
            w += y[i] * X[i]
            b += y[i]


y_predict = predict(X_test)
correct = np.sum(y_predict == y_test)
print ("correct: ", correct, "/", len(y_predict))
acc = np.mean(y_predict == y_test)
print(acc)

