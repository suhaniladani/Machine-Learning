
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


train_df.insert(0, 'Ones', 1)
col_nums = train_df.shape[1]

X = train_df.iloc[:,0:col_nums-1]  
y = train_df.iloc[:,col_nums-1:col_nums] 
y


# In[4]:


IdentitySize = X.shape[1]
IdentityMatrix= np.zeros((IdentitySize, IdentitySize))
np.fill_diagonal(IdentityMatrix, 1)


# In[5]:


a_cont = 1
xTx = X.T.dot(X) + a_cont * IdentityMatrix
XtX = np.linalg.pinv(xTx)
XtX_xT = XtX.dot(X.T)
coefficient = XtX_xT.dot(y)


# In[6]:


coefficient


# In[7]:


test_df.insert(0, 'Ones', 1)
col_nums = test_df.shape[1]
X = test_df.iloc[:,0:col_nums-1]  
y = test_df.iloc[:,col_nums-1:col_nums] 
y_pred = coefficient.T.dot(X.T)

y_pred


# In[8]:


y_pred = y_pred.T 
mse = np.mean((y - y_pred)**2)
mse

