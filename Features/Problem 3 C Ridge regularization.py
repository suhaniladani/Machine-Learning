
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


X_train = pd.read_csv("spam_polluted/train_feature.txt", delim_whitespace=True, header=None) 
X_test = pd.read_csv("spam_polluted/test_feature.txt", delim_whitespace=True, header=None)
y_train = pd.read_csv("spam_polluted/train_label.txt", delim_whitespace=True, header=None)
y_test = pd.read_csv("spam_polluted/test_label.txt", delim_whitespace=True, header=None)


# In[3]:


y_train.columns = ['label']
y_test.columns = ['label']


# In[4]:


m, n = X_train.shape
p, q = X_test.shape
total = m+p


# In[5]:


df = X_train.append(X_test)
df = (df - df.mean()) / (df.max() - df.min())


# In[6]:


X_train = df.iloc[0:m, :]
X_test = df.iloc[m:total, :]


# In[7]:


train_df = X_train.join(y_train)
test_df = X_test.join(y_test)


# In[8]:


col_nums = train_df.shape[1]

train_X = train_df.iloc[:,0:col_nums-1]  
train_y = train_df.iloc[:,col_nums-1:col_nums] 


# In[9]:


X = np.array(train_X)
ones = np.ones(len(X))
X = np.column_stack((ones,X))
y = np.array(train_y)   


# In[10]:


def ridge_regression(X, y, lam):
    

    Xt = np.transpose(X)
    lambda_identity = lam*np.identity(len(Xt))
    inverse_term = np.linalg.inv(np.dot(Xt, X)+lambda_identity)
    w = np.dot(np.dot(inverse_term, Xt), y)
    return w


# In[11]:


# test_df.insert(0, 'Ones', 1)
# col_nums = test_df.shape[1]
# X = test_df.iloc[:,0:col_nums-1]  
# y = test_df.iloc[:,col_nums-1:col_nums] 
# y_pred = coefficient.T.dot(X.T)

# y_pred


# In[12]:


# y_pred = y_pred.T 
# mse = np.mean((y - y_pred)**2)
# mse


# In[13]:


w = ridge_regression(X, y, 0.001)


# In[14]:


w


# In[15]:


y_pred_train = w.T.dot(X.T)
y_pred_train = y_pred_train.T 
mse_train = np.mean((y - y_pred_train)**2)
mse_train


# In[16]:


threshold = 0.4
y_pred_train[y_pred_train < threshold] = 0
y_pred_train[y_pred_train > threshold] = 1


# In[17]:


print("Accuracy:", np.mean(y == y_pred_train)) 


# In[18]:



col_nums = test_df.shape[1]
X_test = test_df.iloc[:,0:col_nums-1]  
y_test = test_df.iloc[:,col_nums-1:col_nums] 

X_test = np.array(X_test)
ones = np.ones(len(X_test))
X_test = np.column_stack((ones,X_test))
y_test = np.array(y_test)

y_pred = w.T.dot(X_test.T)


# In[19]:


y_pred = y_pred.T 
mse = np.mean((y_test - y_pred)**2)
mse


# In[22]:


threshold = 0.4
y_pred[y_pred < threshold] = 0
y_pred[y_pred > threshold] = 1
y_pred.shape


# In[23]:


print("Accuracy:", np.mean(y_test == y_pred)) 


# In[24]:


TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
 
TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
 
FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
 
FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))


# In[25]:


print("TP=", TP, "TN=",TN, "FP=",FP, "FN=",FN)

