
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import numpy as np
import pandas as pd
import random
import math

from scipy.stats import multivariate_normal
import scipy.stats
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("../spambase/spambase.data", names = word_labels, header=None) 
df_norm = df.iloc[:, :-1]
df_norm = (df_norm - df_norm.mean()) / df_norm.std()
df = df_norm.join(df.iloc[:, -1])

data = df


# In[3]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indexes = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    
    return train_df, test_df


# In[4]:


random.seed(0)
train_df, test_df = train_test_split(df, 0.20)


# In[5]:


X_train = np.array(train_df.iloc[: , :-1])
y_train = np.array(train_df.iloc[: , -1])

X_test = np.array(test_df.iloc[: , :-1])
y_test = np.array(test_df.iloc[: , -1])


# In[6]:


num_spam = df['label'][df['label'] == 1].count()
num_non_spame = df['label'][df['label'] == 0].count()
total = len(df)

print('Spam:',num_spam)
print('Non-spam ',num_non_spame)
print('Total: ',total)


# In[7]:


prob_spam = num_spam/total
print('Probability spam: ',prob_spam)

prob_non_spam = num_non_spame/total
print('Probability non-spam: ',prob_non_spam)

# def class_prob(cls):
#     n = len([item for item in (y_train) if item == cls])
#     d = len(y_train)
#     return n * 1.0 /d


# In[8]:


data_mean = df.groupby('label').mean()

data_variance = df.groupby('label').var()*(1/6)


# In[9]:


sigma = 1
def gauss_kernel(x1,x2):
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-sigma * distance)


# In[10]:


y_pred = []


# In[11]:


unique_classes = np.unique(y_train)

unique_classes


# In[12]:


def llh_func(X):
    llh = np.zeros((X.shape[0], np.size(unique_classes)))
    llh[:,0] += np.log(prob_non_spam)
    llh[:,1] += np.log(prob_spam)
    #print llh
    for i, x in enumerate(X):
        for cls in unique_classes:
            prob = 0
            for j, f in enumerate(X_train):
                if y_train[j] != cls: continue
                prob += gauss_kernel(x, f)
            c = np.where(unique_classes == cls)
            #print prob
            llh[i, c] += np.log(prob)
    #print llh
    return llh.T


# In[13]:


def predict(X):
    llh = llh_func(X)
#     print(llh)
    return unique_classes[np.argmax(llh, axis=0)]


# In[14]:


y_pred = predict(X_test)


# In[15]:


np.mean(y_pred == y_test)

