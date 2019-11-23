
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pandas import get_dummies
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("WineData/wine.txt", header=None) 
df.head()


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


x_input=train.loc[:,['f1','f2','f3','f4','f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']]
y_input=train.loc[:, ['label1', 'label2', 'label3']]


# In[10]:


X_test=test.loc[:,['f1','f2','f3','f4','f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']]
y_test=test.loc[:, ['label1', 'label2', 'label3']]


# In[11]:


y_input.shape[1]


# In[12]:


x_p=tf.placeholder(tf.float32,shape=[None,x_input.shape[1]])
y_p=tf.placeholder(tf.float32,shape=[None, y_input.shape[1]])


# In[13]:


num_features = 13
num_labels = 3
num_hidden = 100


# In[15]:


W1=tf.Variable(tf.random_normal([num_features,num_hidden]))
b1=tf.Variable(tf.zeros([num_hidden]))


# In[16]:


W2 = tf.Variable(tf.random_normal([num_hidden, num_labels]))
b2 = tf.Variable(tf.zeros([num_labels]))


# In[17]:



z1 = tf.matmul(x_p , W1 ) + b1
a1 = tf.nn.sigmoid(z1)
z2 = tf.matmul(a1, W2) + b2

y = tf.nn.softmax(z2)

loss = tf.reduce_mean(-tf.reduce_sum(y_p * tf.log(y), reduction_indices=[1]))


optimizer_train = tf.train.AdamOptimizer(0.005).minimize(loss)

y_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_p,1))
accuracy = tf.reduce_mean(tf.cast(y_pred, tf.float32))


# In[18]:



sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)
epochs=10000


# In[19]:


x_input


# In[20]:


b = y_input.values
b


# In[21]:


for step in range(2,epochs):
    _, c=sess.run([optimizer_train,loss], feed_dict={x: x_input, y_:[t for t in y_input.values]})
    
    if step%500==0:
        print (c)


# In[22]:


print ("Accuracy is " , sess.run(accuracy,feed_dict={x_p: x_input, y_p:[t for t in y_input.values]}))
print ("Accuracy is " , sess.run(accuracy,feed_dict={x_p: X_test, y_p:[t for t in y_test.values]}))

