
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.optimize as sp
import random
from pprint import pprint


# In[2]:


X_train = pd.read_csv("spam_polluted/train_feature.txt", delim_whitespace=True, header=None) 
X_test = pd.read_csv("spam_polluted/test_feature.txt", delim_whitespace=True, header=None)
y_train = pd.read_csv("spam_polluted/train_label.txt", delim_whitespace=True, header=None)
y_test = pd.read_csv("spam_polluted/test_label.txt", delim_whitespace=True, header=None)


# In[3]:


X_test.shape


# In[4]:


y_train.columns = ['label']
y_test.columns = ['label']


# In[5]:


m, n = X_train.shape
p, q = X_test.shape
total = m+p


# In[6]:


# train_df = X_train.join(y_train)
# test_df = X_test.join(y_test)


# In[7]:


df = X_train.append(X_test)


# In[8]:


df = (df - df.mean()) / (df.max() - df.min())


# In[9]:


X_train = df.iloc[0:m, :]
X_test = df.iloc[m:total, :]


# In[10]:


train_df = X_train.join(y_train)
test_df = X_test.join(y_test)


# In[11]:


def sigmoid(theta, X): 

    return 1.0/(1 + np.exp(-np.dot(X, theta.T))) 


# In[12]:


def gradient(theta, X, y): 

    hypothesis_cost = sigmoid(theta, X) - y.reshape(X.shape[0], -1) 
    gradient = np.dot(hypothesis_cost.T, X) 
    return gradient 
  


# In[13]:


def cost_function(theta, X, y): 

    hypothesis = sigmoid(theta, X) 
    y = np.squeeze(y) 
    term1 = y * np.log(hypothesis) 
    term2 = (1 - y) * np.log(1 - hypothesis) 
    calculate_cost = -term1 - term2 
    return np.mean(calculate_cost) 
  


# In[14]:


def gradient_descent(X, y, theta, alpha=.001, max_cost=.001): 

    cost = cost_function(theta, X, y) 
    change_cost = np.inf
      
    while(change_cost > max_cost): 
        old_cost = cost 
        theta = theta - (alpha * gradient(theta, X, y)) 
        cost = c(theta, X, y) 
        change_cost = old_cost - cost 
      
    return theta
  


# In[15]:


def predict_values(theta, X): 

    pred_prob = sigmoid(theta, X) 
    pred_value = np.where(pred_prob >= .5, 1, 0) 
    return np.squeeze(pred_value) 


# In[16]:


X = train_df.iloc[:, :-1]
X = np.array(X)


X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X)) 

 
y = train_df.iloc[:, -1] 
y = np.array(y)

theta = np.matrix(np.zeros(X.shape[1])) 


theta = gradient_descent(X, y, theta) 


y_pred = predict_values(theta, X) 

print("Accuracy:", np.mean(y == y_pred)) 


# In[17]:


X_test = test_df.iloc[:, :-1]
X_test = np.array(X_test)


X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test)) 


y_test = test_df.iloc[:, -1] 
y_test = np.array(y_test)


# In[18]:


y_pred_test = predict_values(theta, X_test) 


# In[19]:


print("Accuracy:", np.mean(y_test == y_pred_test)) 


# In[20]:


thresholds = np.linspace(2,-2,105)

ROC = np.zeros((105,2))

for i in range(105):
    t = thresholds[i]

    TP_t = np.logical_and( y_pred_test > t, y_test==1 ).sum()
    TN_t = np.logical_and( y_pred_test <=t, y_test==0 ).sum()
    FP_t = np.logical_and( y_pred_test > t, y_test==0 ).sum()
    FN_t = np.logical_and( y_pred_test <=t, y_test==1 ).sum()

    FPR_t = FP_t / float(FP_t + TN_t)
    ROC[i,0] = FPR_t

    TPR_t = TP_t / float(TP_t + FN_t)
    ROC[i,1] = TPR_t

# Plot the ROC curve.
fig = plt.figure(figsize=(6,6))
plt.plot(ROC[:,0], ROC[:,1], lw=2)
plt.xlabel('$FPR(t)$')
plt.ylabel('$TPR(t)$')
plt.grid()


# In[21]:


AUC = 0.
for i in range(100):
    AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
AUC *= 0.5
AUC

