
# coding: utf-8

# ## Problem 2 (B)

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[2]:


train_feature = pd.read_csv("spam_polluted/train_feature.txt", delim_whitespace=True, header=None) 
test_feature = pd.read_csv("spam_polluted/test_feature.txt", delim_whitespace=True, header=None) 
train_label = pd.read_csv("spam_polluted/train_label.txt", delim_whitespace=True, header=None) 
test_label = pd.read_csv("spam_polluted/test_label.txt", delim_whitespace=True, header=None) 


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


train_features = principalDf.iloc[0:4140]
test_features = principalDf.iloc[4140:]


# In[9]:


train_label = train_label.rename(columns={0: "target"})


# In[10]:


finalDf = pd.concat([train_features, train_label], axis = 1)


# In[11]:


finalDf.head()


# In[12]:


clf = GaussianNB()
clf.fit(train_features, train_label)
target_pred = clf.predict(test_features)


# In[13]:


accuracy_score(test_label, target_pred, normalize = True)


# ## Problem 3 (B)

# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


lr2 = LogisticRegression(penalty='l2')
lr2.fit(train_feature, train_label)


# In[16]:


y_pred = lr2.predict(test_feature)
y_pred = pd.DataFrame(data = y_pred)


# In[17]:


print("Accuracy:", np.mean(test_label == y_pred))


# In[18]:


lr1 = LogisticRegression(penalty='l1')
lr1.fit(train_feature, train_label)


# In[19]:


y_pred1 = lr1.predict(test_feature)
y_pred1 = pd.DataFrame(data = y_pred)


# In[20]:


print("Accuracy:", np.mean(test_label == y_pred1))

