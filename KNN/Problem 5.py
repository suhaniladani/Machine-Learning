
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import random
from collections import defaultdict


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


y_test


# In[7]:


n_neighbors=5

X = X_train
y = y_train


# In[8]:


def calc_dist(x1, x2):
    return sum(abs(x1 - x2))


# In[9]:


def calculate_w(distances):
    matches = [(1, y) for d, y in distances if d == 0]
    return matches if matches else [(1/d, y) for d, y in distances]


# In[10]:


def predict_single(test):
    distances = sorted((calc_dist(x, test), y) for x, y in zip(X, y))
    w = calculate_w(distances[:n_neighbors])
    w_by_class = defaultdict(list)
    for d, c in w:
        w_by_class[c].append(d)
    return max((sum(val), key) for key, val in w_by_class.items())[1]


# In[11]:


def predict(X):
    return [predict_single(i) for i in X]


# In[ ]:


y_pred = predict(X_test)


# In[ ]:


acc = np.mean(y_pred == y_test)


# In[ ]:


acc

