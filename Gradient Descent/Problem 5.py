
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("WineData/wine.txt", header=None) 


# In[3]:


train = pd.read_csv("WineData/train_wine.csv", header=None) 
test = pd.read_csv("WineData/test_wine.csv", header=None) 
test.head()


# In[4]:


df = train.append(test)
df.tail()


# In[5]:


df_norm = df[[1,2,3,4,5,6,7,8,9,10,11,12,13]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.head()


# In[6]:


target = df[[0]]
df = pd.concat([df_norm, target], axis=1)
df.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'label']
label_cols = pd.get_dummies(df.label)
df = pd.concat([df_norm, label_cols], axis=1)
df.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'label1', 'label2', 'label3']
df.head()


# In[7]:


train = df.iloc[:151, :]
train.head()


# In[8]:


test = df.iloc[151: , :]
test.head()


# In[9]:


def sigmoid(z):
    return 1/(1 + np.exp(-(z)))


# In[10]:


def softmax(z):
    e = np.exp(z - np.sum(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# In[11]:


def cross_entropy(y, y_pred):
    m = y.shape[0]
    cost = -(1.0/m) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    return cost


# In[12]:


X = train.values[:,:13]
X[:5]


# In[13]:


num_inputs = len(X[0])
hidden_layer_neurons = 5
np.random.seed(4)
w1 = np.random.random((num_inputs, hidden_layer_neurons)) - 1
w1


# In[14]:


y = np.array(train.iloc[:, 13:])
y


# In[15]:


num_outputs = len(y[0])
w2 = np.random.random((hidden_layer_neurons, num_outputs)) - 1
w2


# In[16]:


alpha = 0.2 
for epoch in range(50000):
    z1 = np.dot(X, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    a2 = softmax(z2)
    loss = cross_entropy(y, a2)
    a2_delta = (y - a2)*(a2 * (1-a2))
    a1_delta = a2_delta.dot(w2.T) * (a1 * (1-a1))
    w2 += a1.T.dot(a2_delta) * alpha
    w1 += X.T.dot(a1_delta) * alpha
print('Loss:', loss)


# In[17]:


z1 = np.dot(X, w1)
a1 = sigmoid(z1)
z2 = np.dot(a1, w2)
a2 = softmax(z2)

np.round(a2,3)


# In[18]:


y_pred = np.argmax(a2, axis=1)
y_result = np.argmax(y, axis = 1)
result = y_pred == np.argmax(y, axis=1)
correct = np.sum(result)/len(result)

print('Correct:',sum(result),'/',len(result))
print('Accuracy:', (correct*100),'%')


# In[19]:


X = test.values[:,:13]
y = np.array(test.iloc[:, 13:])

z1 = np.dot(X, w1)
a1 = sigmoid(z1)
z2 = np.dot(a1, w2)
a2 = softmax(z2)

np.round(a2,3)


# In[20]:


y_pred = np.argmax(a2, axis=1)
y_result = np.argmax(y, axis = 1)
result = y_pred == np.argmax(y, axis=1)
correct = np.sum(result)/len(result)

print('Correct:',sum(result),'/',len(result))
print( 'Accuracy:', (correct*100),'%')


# In[21]:


y_pred

