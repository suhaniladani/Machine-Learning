
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import pinv
import random
import math


# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("spambase/spambase.data", names = word_labels, header=None) 
df_norm = df.iloc[:, :-1]
df_norm = (df_norm - df_norm.mean()) / df_norm.std()
df = df_norm.join(df.iloc[:, -1])


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


# In[5]:


X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]
m,n = np.shape(X)
x0 = np.ones((m,1))
X = np.c_[x0,X]


# In[6]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[8]:


def hessian(X,h):
    return np.dot(X.T,(np.diag(h*(1-h)).dot(X))) 


# In[7]:


def cost_func(h, y):
    return np.sum((y * np.log(h) + (1 - y) * np.log(1 - h)))


# In[9]:


def Newton_logistic(X,y):
    w = np.zeros(X.shape[1])
    cost=0
    for i in range(1000):

        hypothesis = sigmoid(np.dot(X, w))
        gradient = np.dot(X.T, (y-hypothesis)) 
        hessian_inv=pinv(hessian(X,hypothesis))
        w+=hessian_inv.dot(gradient) 
        prev_cost = cost
        cost=cost_func(hypothesis,y) 

        if cost-prev_cost==0:
            break;

    return w


# In[10]:


X


# In[11]:


w = Newton_logistic(X,y)
w


# In[12]:


y_pred = w.T.dot(X.T)


# In[13]:


y_pred


# In[14]:


threshold = 0.0
y_pred = []
for i in range(0, len(X-1)):
    val = np.dot(X[i], w)   

    if val > threshold:                               
        y_predict = 1                               
    else:                                   
        y_predict = 0
    y_pred.append(y_predict)


# In[15]:


print("Accuracy:", np.mean(y == y_pred)) 


# In[16]:


X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
m,n = np.shape(X_test)
x0 = np.ones((m,1))
X_test = np.c_[x0,X_test]


# In[17]:


threshold = 0.0
y_pred_test = []
for i in range(0, len(X_test-1)):
    val = np.dot(X_test[i], w)   

    if val > threshold:                               
        y_predict = 1                               
    else:                                   
        y_predict = 0
    y_pred_test.append(y_predict)


# In[18]:


print("Accuracy:", np.mean(y_test == y_pred_test)) 

