
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import random
import math
import sys


# In[18]:


df = pd.read_csv("CRX/crx.data", delimiter='\t', header=None) 
df.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'label']
df.head()


# In[19]:


X = pd.get_dummies(df.drop('label', axis=1))


# In[20]:


X.head()


# In[21]:


df['label'].replace('-', -1,inplace=True)
df['label'].replace('+', 1,inplace=True)
df.head()


# In[24]:


y = df.iloc[:, -1]
df = X.join(y)


# In[25]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indexes = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    
    return train_df, test_df


# In[26]:


random.seed(3)
train_df, test_df = train_test_split(df, 0.20)


# In[27]:


X_train = train_df.iloc[: , :-1]
y_train = train_df.iloc[: , -1]

X_test = test_df.iloc[: , :-1]
y_test = test_df.iloc[: , -1]


# In[28]:


# Determines if sample shall be classified as -1 or 1 given threshold
def make_clf_dict():
    clf_dict = {'label': 1, 'column': None, 'threshold': None, 'alpha': None}
    return clf_dict


# In[29]:


def predict(X):
    n_samples = np.shape(X)[0]
    y_pred = np.zeros((n_samples, 1))
    for clf in classifiers:
        predictions = np.ones(np.shape(y_pred))
        index_neg = (clf['label'] * X[clf['column']] < clf['label'] * clf['threshold'])
        predictions[index_neg] = -1
        y_pred += clf['alpha'] * predictions

    y_pred = np.sign(y_pred).flatten()

    return y_pred



# In[30]:


iters = 15

def adaboost(X, y):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))

    classifiers = []

    for i in range(iters):
       
      clf = make_clf_dict()

        min_err = float('inf')

        for column in X_train.columns:
            feature_values = np.expand_dims(X[column], axis=1)
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                p = 1

                prediction = np.ones(np.shape(y))
                prediction[X[column] < threshold] = -1
                error = sum(w[y != prediction])

                if error > 0.5:
                    error = 1 - error
                    p = -1

                if error < min_err:
                    clf['label'] = p
                    clf['threshold'] = threshold
                    clf['column'] = column
                    min_err = error
        clf['alpha'] = 0.5 * math.log((1.0 - min_err) / (min_err + 1e-10))
        predictions = np.ones(np.shape(y))
        index_neg = (clf['label'] * X[clf['column']] < clf['label'] * clf['threshold'])
        predictions[index_neg] = -1
        w *= np.exp(-clf['alpha'] * y * predictions)
        w /= np.sum(w)

        classifiers.append(clf)
    return classifiers


# In[31]:


classifiers = adaboost(X_train, y_train)
y_pred = predict(X_test)
y_train_pred = predict(X_train)



# In[32]:


np.mean(np.array(y_test) == y_pred)

