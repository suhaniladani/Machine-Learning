
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


# In[3]:


data = pd.read_csv('../spambase/spambase.data', header=None)
data.rename(columns={57:'label'}, inplace=True)


# In[5]:


data.head()


# In[6]:


y = data.pop('label')
X = data


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[7]:


clf = svm.SVC(gamma='scale', kernel='rbf')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", np.mean(y_test == y_pred))


# In[8]:


clf1 = svm.SVC(gamma='scale', kernel='poly')

clf1.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)

print("Accuracy:", np.mean(y_test == y_pred1))


# In[9]:


clf2 = svm.SVC(gamma='scale', kernel='linear')

clf2.fit(X_train, y_train)

y_pred2 = clf2.predict(X_test)

print("Accuracy:", np.mean(y_test == y_pred2))


# In[10]:


clf3 = svm.SVC(gamma='scale', kernel='sigmoid')

clf3.fit(X_train, y_train)

y_pred3 = clf3.predict(X_test)

print("Accuracy:", np.mean(y_test == y_pred3))


# In[8]:


clf4 = svm.SVC(random_state=0, gamma=.01, C=1)

clf4.fit(X_train, y_train)

y_pred4 = clf4.predict(X_test)

print("Accuracy:", np.mean(y_test == y_pred4))

