
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import numpy as np
import pandas as pd
import random
import math


# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("../spambase/spambase.data", names = word_labels, header=None) 
# df_norm = df.iloc[:, :-1]
# df_norm = (df_norm - df_norm.mean()) / df_norm.std()
# df = df_norm.join(df.iloc[:, -1])


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


def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


# In[7]:


r = 2.5

def count_class(neighbor_labels):
    counts = np.bincount(neighbor_labels.astype('int'))
    return counts.argmax()

def predict(X_test, X_train, y_train):
    y_pred = np.empty(X_test.shape[0])
    for i, test_row in enumerate(X_test):
        p = [euclidean_distance(test_row, x) for x in X_train]
        indexed = np.argsort(p)
        list(filter(lambda i: p[i] < r, indexed))
        knn = np.array([y_train[i] for i in indexed])
        y_pred[i] = count_class(knn)

    return y_pred


# In[8]:


y_pred = predict(X_test, X_train, y_train)
y_pred = y_pred.astype(int)


# In[9]:


# acc = np.mean(y_pred == y_test)


# In[10]:


# acc


# In[11]:


correct = 0
for i in range(y_pred.shape[0]):
    if(y_pred[i] == y_test[i]):
        correct += 1


# In[12]:


acc = correct/y_pred.shape[0]


# In[2]:


acc

