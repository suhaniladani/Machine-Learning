
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import pinv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random


# In[2]:


word_labels = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "label"]
train_df = pd.read_csv("HousingData/housing_train.txt", delim_whitespace=True, names = word_labels, header=None) 
test_df = pd.read_csv("HousingData/housing_test.txt", delim_whitespace=True, names = word_labels, header=None) 


# In[3]:


col_nums = train_df.shape[1]

train_X = train_df.iloc[:,0:col_nums-1]  
train_y = train_df.iloc[:,col_nums-1:col_nums] 


# In[4]:


X = np.array(train_X)
ones = np.ones(len(X))
X = np.column_stack((ones,X))
y = np.array(train_y)


# In[5]:


def ridge_regression(X, y, lam):
      
    Xt = np.transpose(X)
    lambda_identity = lam*np.identity(len(Xt))
    inverse_term = pinv(np.dot(Xt, X)+lambda_identity)
    w = np.dot(np.dot(inverse_term, Xt), y)
    return w


# In[6]:


# test_df.insert(0, 'Ones', 1)
# col_nums = test_df.shape[1]
# X = test_df.iloc[:,0:col_nums-1]  
# y = test_df.iloc[:,col_nums-1:col_nums] 
# y_pred = coefficient.T.dot(X.T)

# y_pred


# In[7]:


# y_pred = y_pred.T 
# mse = np.mean((y - y_pred)**2)
# mse


# In[8]:


w = ridge_regression(X, y, 1)


# In[9]:


w


# In[10]:


y_pred_train = w.T.dot(X.T)


# In[11]:


y_pred_train = y_pred_train.T 
mse = np.mean((y - y_pred_train)**2)
mse


# In[12]:



col_nums = test_df.shape[1]
X_test = test_df.iloc[:,0:col_nums-1]  
y_test = test_df.iloc[:,col_nums-1:col_nums] 

X_test = np.array(X_test)
ones = np.ones(len(X_test))
X_test = np.column_stack((ones,X_test))
y_test = np.array(y_test)

y_pred = w.T.dot(X_test.T)

y_pred


# In[13]:


y_pred = y_pred.T 
mse = np.mean((y_test - y_pred)**2)
mse

