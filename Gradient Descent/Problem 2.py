
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import get_dummies
import tensorflow as tf


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


X_data = df.iloc[:,:8]


# In[9]:


y_data = df.iloc[:, 8:]
 


# In[10]:


x_p =tf.placeholder(tf.float32,shape=[None,X_data.shape[1]])
y_p=tf.placeholder(tf.float32,shape=[None, y_data.shape[1]])


# In[11]:


num_features = 8
num_labels = 8
num_hidden = 3


# In[12]:


W1=tf.Variable(tf.random_normal([num_features,num_hidden]))
b1=tf.Variable(tf.zeros([num_hidden]))


# In[13]:


W2 = tf.Variable(tf.random_normal([num_hidden, num_labels]))
b2 = tf.Variable(tf.zeros([num_labels]))


# In[14]:


z1 = tf.matmul(x_p , W1 ) + b1
a1 = tf.nn.sigmoid(z1)
z2 = tf.matmul(a1, W2) + b2

y = tf.nn.softmax(z2)


# In[15]:


loss = tf.reduce_mean(tf.square(tf.subtract(y, y_p)))

optimizer_train = tf.train.AdamOptimizer(0.005).minimize(loss)

y_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_p,1))
accuracy = tf.reduce_mean(tf.cast(y_pred, tf.float32))


# In[16]:


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
epochs=10000


# In[17]:


for iteration in range(2,epochs):
    _, l=sess.run([optimizer_train,loss], feed_dict={x_p: X_data, y_p :[t for t in y_data.values]})
    
    if iteration%500==0:
        print (l)


# In[18]:


print ("Accuracy:" , sess.run(accuracy,feed_dict={x_p: X_data, y_p:[t for t in y_data.values]}))

