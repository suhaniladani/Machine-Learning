
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import random

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[2]:


train_feature = pd.read_csv("spam_polluted/train_feature.txt", delim_whitespace=True, header=None) 
test_feature = pd.read_csv("spam_polluted/test_feature.txt", delim_whitespace=True, header=None) 
train_label = pd.read_csv("spam_polluted/train_label.txt", delim_whitespace=True, header=None) 
y_test = pd.read_csv("spam_polluted/test_label.txt", delim_whitespace=True, header=None) 


# In[3]:


WholeDf = pd.concat([train_feature, test_feature])


# In[4]:


WholeDf.head()


# In[5]:


pca = PCA(n_components=101)


# In[6]:


principalComponents = pca.fit_transform(WholeDf)
principalDf = pd.DataFrame(data = principalComponents)


# In[7]:


principalDf.head()


# In[8]:


X_train = principalDf.iloc[0:4140]
X_test = principalDf.iloc[4140:]


# In[9]:


y_train = train_label.rename(columns={0: "label"})


# In[10]:


finalDf = pd.concat([X_train, y_train], axis = 1)


# In[11]:


num_spam = finalDf['label'][finalDf['label'] == 1].count()
num_non_spame = finalDf['label'][finalDf['label'] == 0].count()
total = len(finalDf)

print('Spam:',num_spam)
print('Non-spam ',num_non_spame)
print('Total: ',total)


# In[12]:


prob_spam = num_spam/total
print('Probability spam: ',prob_spam)

prob_non_spam = num_non_spame/total
print('Probability non-spam: ',prob_non_spam)


# In[13]:



data_mean = finalDf.groupby('label').mean()

data_variance = finalDf.groupby('label').var()*(1/6)


# In[14]:


def prob_x_y(x, mean_y, variance_y):
    prob = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return prob


# In[15]:


y_pred = []


# In[16]:


for row in range(0,len(X_train)):
        prod_0 = prob_non_spam
        prod_1 = prob_spam
        for col in X_train.columns:   
            prod_0 *= prob_x_y(X_train[col].iloc[row], data_mean[col][0], data_variance[col][0])
            prod_1 *= prob_x_y(X_train[col].iloc[row], data_mean[col][1], data_variance[col][1])
    

        if prod_0 > prod_1:
            y_pred.append(0)
        else:
            y_pred.append(1)


# In[17]:


np.array(y_train)


# In[18]:


np.mean(y_pred== y_train)

