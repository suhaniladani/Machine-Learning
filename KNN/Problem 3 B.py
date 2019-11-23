
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg
from __future__ import print_function, division
import pandas as pd
import random
import math


# In[2]:


df = pd.read_csv("twoSpirals.txt", delim_whitespace=True, header=None) 
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


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


# In[7]:


def gaussian_kernel(x, y, sigma=2.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


# In[8]:


def project(X):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, svv_y, svv in zip(alpha, sv_y, sv_x):
            s += a * svv_y * gaussian_kernel(X[i], svv)
        y_predict[i] = s
    return y_predict


# In[9]:


def predict(X):
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    return np.sign(project(X))


# In[10]:


T=3
X = X_train
y = y_train

n_samples, n_features = X.shape
alpha = np.zeros(n_samples, dtype=np.float64)


K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i,j] = gaussian_kernel(X[i], X[j])

for t in range(T):
    for i in range(n_samples):
        if np.sign(np.sum(K[:,i] * alpha * y)) != y[i]:
            alpha[i] += 1.0

sv = alpha > 1e-5
ind = np.arange(len(alpha))[sv]
alpha = alpha[sv]
sv_x = X[sv]
# print (y)
sv_y = y[sv]


y_predict = predict(X_test)
correct = np.mean(y_predict == y_test)
print (correct)

