
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import pinv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random


# In[2]:


df = pd.read_csv("PerceptronData/perceptronData.txt", delim_whitespace=True, header=None) 
df.tail()


# In[3]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indices = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[4]:


random.seed(0)
train_df, test_df = train_test_split(df, test_size=0.20)
x_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
x_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]


# In[5]:


X = np.array(x_train)
ones = np.ones(len(X))*(-1)
X = np.column_stack((X, ones))
Y = np.array(y_train)


# In[6]:



def perceptron_algo(X, Y):

    w = np.zeros(len(X[0]))
    alpha = 0.1
    num_iterations = 30

    for iteration in range(num_iterations):
        total_error = 0
        for i, x in enumerate(X):
            missclassification = (np.dot(X[i], w)*Y[i])
            if missclassification <= 0:
                total_error += missclassification
                w = w + alpha*X[i]*Y[i]
    
    return w


# In[7]:


w = perceptron_algo(X, Y)


# In[8]:


w


# In[9]:


X = np.array(x_test)
ones = np.ones(len(X))*(-1)
X = np.column_stack((X, ones))
Y = np.array(y_test)


# In[10]:


threshold = 0.0
alpha = 0.1

y_pred = []
for i in range(0, len(X-1)):
    val = np.dot(X[i], w)   

    if val > threshold:                               
        y_predict = 1                               
    else:                                   
        y_predict = -1
    y_pred.append(y_predict)


# In[11]:


y_pred


# In[12]:


print("Accuracy:", np.mean(y_test == y_pred)) 


# In[13]:


len(y_test)

