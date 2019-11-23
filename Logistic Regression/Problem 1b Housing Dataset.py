
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# In[2]:


word_labels = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "label"]
train_df = pd.read_csv("HousingData/housing_train.txt", delim_whitespace=True, names = word_labels, header=None) 
test_df = pd.read_csv("HousingData/housing_test.txt", delim_whitespace=True, names = word_labels, header=None) 


# In[3]:


theta_len = len(train_df.columns)
theta_len


# In[4]:



x = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 


# In[5]:


alpha = 0.002 
iterations = 2000 
m = y.size
np.random.seed(123) 
theta = np.random.rand(theta_len) 


def gradient_descent(x, y, theta, iterations, alpha):
    costs = []
    thetas = [theta]
    for i in range(iterations):
        hypothesis = np.dot(x, theta)
        error = hypothesis - y
        cost = 1/(2*m) * np.dot(error.T, error)
        costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        thetas.append(theta)
        
    return thetas, costs


thetas, costs = gradient_descent(x, y, theta, iterations, alpha)
theta = thetas[-1]


# In[6]:


theta


# In[7]:


y_pred_train = theta.T.dot(x.T)


# In[8]:


y_pred_train = y_pred_train.T 
mse = np.mean((y - y_pred_train)**2)
mse


# In[9]:


x_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

x_test = (x_test - x_test.mean()) / x_test.std()
x_test = np.c_[np.ones(x_test.shape[0]), x_test] 


# In[10]:


y_pred = theta.T.dot(x_test.T)


# In[11]:


y_pred = y_pred.T 
mse_test = np.mean((y_test - y_pred)**2)
mse_test

