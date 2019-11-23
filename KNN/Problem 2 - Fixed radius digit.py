
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import numpy as np
import pandas as pd
import random
import math
from scipy.spatial.distance import cosine


# In[2]:


# word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
#                 "order", "mail", "receive", "will", "people", "report", "addresses",
#                 "free", "business", "email", "you", "credit", "your", "font", "000",
#                 "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
#                 "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
#                 "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
#               "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
# df = pd.read_csv("../spambase/spambase.data", names = word_labels, header=None) 
# df_norm = df.iloc[:, :-1]
# df_norm = (df_norm - df_norm.mean()) / df_norm.std()
# df = df_norm.join(df.iloc[:, -1])
data = np.load('../mnist_test_data')
df_X = pd.read_csv("../Haar_Features/X_df.csv")
df_y = pd.read_csv("../Haar_Features/y_df.csv")

X = np.array(df_X)
y = np.array(df_y)
y = np.reshape(y, len(y))


# In[3]:


data[0]


# In[4]:


from sklearn.model_selection import train_test_split
# def train_test_split(df, test_size):
    
#     if isinstance(test_size, float):
#         test_size = round(test_size * len(df))

#     index_list = df.index.tolist()
#     test_indexes = random.sample(population=index_list, k=test_size)

#     test_df = df.loc[test_indexes]
#     train_df = df.drop(test_indexes)
    
#     return train_df, test_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[5]:


# random.seed(0)
# train_df, test_df = train_test_split(df, 0.20)


# In[6]:


# X_train = np.array(train_df.iloc[: , :-1])
# y_train = np.array(train_df.iloc[: , -1])

# X_test = np.array(test_df.iloc[: , :-1])
# y_test = np.array(test_df.iloc[: , -1])


# In[7]:


def cosine_distance(p1, p2):
    return cosine(p1, p2)


# In[8]:


r = 0.83

def count_class(neighbor_labels):
    counts = np.bincount(neighbor_labels.astype('int'))
    return counts.argmax()

def predict(X_test, X_train, y_train):
    y_pred = np.empty(X_test.shape[0])
    for i, test_sample in enumerate(X_test):
        p = [cosine_distance(test_sample, x) for x in X_train]
        indexes = np.argsort(p)
        list(filter(lambda i: p[i] < r, indexes))
        knn = np.array([y_train[i] for i in indexes])
        y_pred[i] = count_class(knn)

    return y_pred


# In[9]:


# np.reshape(X_train, len(X_train))
y_pred = predict(X_test, X_train, y_train)

y_pred = y_pred.astype(int)


# In[10]:


# acc = np.mean(y_pred == y_test)


# In[11]:


# acc


# In[12]:


correct = 0
for i in range(y_pred.shape[0]):
    if(y_pred[i] == y_test[i]):
        correct += 1


# In[13]:


acc = correct/y_pred.shape[0]


# In[14]:


acc

