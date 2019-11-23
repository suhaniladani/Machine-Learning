
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import random


# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("../spambase/spambase.data", names = word_labels, header=None) 


# In[3]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_ind = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_ind]
    train_df = df.drop(test_ind)
    
    return train_df, test_df


# In[4]:


random.seed(0)
train_df, test_df = train_test_split(df, 0.20)


# In[5]:


from random import randrange
def make_kfolds(df, kfolds):
    data_folds = list()
    for i in range(kfolds):
        data_folds.append(pd.DataFrame())
    counter = 0
    for i in range(len(df)):
        if counter >= kfolds:
            counter = 0
        data_folds[counter] = data_folds[counter].append(df[i:i+1])
        counter += 1
    return data_folds


# In[6]:


data_folds = make_kfolds(df, 10)
data_folds[1].head()


# In[7]:


X = np.array(train_df.iloc[:,:-1])
y = np.array(train_df.iloc[:, -1])


# In[8]:


data_number, feature_number = X.shape
unique_cls = np.unique(y)
num_unique_cls = len(unique_cls)


# In[9]:


def fit(X, y):
    phi = np.zeros((num_unique_cls, 1))
    means = np.zeros((num_unique_cls, feature_number))
    Cov_mat = 0
    for i in range(num_unique_cls):
        ind = np.flatnonzero(y == unique_cls[i])
        print (ind)
        phi[i] = len(ind) / data_number
        means[i] = np.mean(X[ind], axis=0)
        Cov_mat += np.cov(X[ind].T) * (len(ind) - 1)

    Cov_mat /= data_number
    return phi, means, Cov_mat


# In[10]:


def predict(phi, means, Cov_mat, X):
    pdf = lambda mean: multivariate_normal.pdf(X, mean=mean, cov=Cov_mat)
    y_probs = np.apply_along_axis(pdf, 1, means) * phi

    return unique_cls[np.argmax(y_probs, axis=0)]


# In[11]:


phi, means, Cov_mat  = fit(X, y)


# In[12]:


y_pred = predict(phi, means, Cov_mat, X)


# In[13]:


accuracy = np.mean(y_pred == y)


# In[14]:


np.mean(y == y_pred)


# In[15]:


Cov_mat.shape


# In[16]:


phi


# In[18]:


means.shape

