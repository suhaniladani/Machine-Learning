
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


a = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8]])


# In[3]:


df = pd.DataFrame(a.reshape(8,2), columns = list("xy"))


# In[4]:


x = pd.get_dummies(df.x)


# In[5]:


y = pd.get_dummies(df.y)


# In[6]:


df = pd.concat([x,y], axis = 1)


# In[7]:


df


# In[8]:


def sigmoid(z):
    return 1/(1 + np.exp(-(z)))


# In[9]:


def cross_entropy(y, y_pred):
    m = y.shape[0]
    cost = -(1.0/m) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    return cost


# In[10]:


X = df.values[:,:8]
X


# In[11]:


y


# In[12]:


num_inputs = len(X[0])
num_hidden = 3
np.random.seed(4)
w1 = 2*np.random.random((num_inputs, num_hidden)) - 1
w1


# In[13]:


y = np.array(df.iloc[:, 8:])
y


# In[14]:


num_outputs = len(y[0])
w2 = np.random.random((num_hidden, num_outputs)) - 1
w2


# In[15]:


alpha = 0.2 
w1_val = []
w2_val = []
for epoch in range(1000):
    z1 = np.dot(X, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)
    loss = cross_entropy(y, a2)
    a2_delta = (y - a2)*(a2 * (1-a2))
    a1_delta = a2_delta.dot(w2.T) * (a1 * (1-a1))
    w2 += a1.T.dot(a2_delta) * alpha
    w2_val.append(w2)
    w1 += X.T.dot(a1_delta) * alpha
    w1_val.append(w1)
print('Loss:', loss)


# In[16]:


y_pred = np.argmax(a2, axis=1) 
y_result = np.argmax(y, axis = 1)
result = y_pred == np.argmax(y, axis=1)
correct = np.sum(result)/len(result)

print('Correct:',sum(result),'/',len(result))
print('Accuracy:', (correct*100),'%')


# In[17]:


w2_val

