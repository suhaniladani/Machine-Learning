
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import numpy as np
import pandas as pd
import random
import math


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


sigma = 0.7
def RBF(x1,x2):
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-sigma * distance)


# In[7]:


def count_class(neighbor_labels):
    counts = np.bincount(neighbor_labels.astype('int'))
    return counts.argmax()

def predict(X_test, X_train, y_train, k):
    y_pred = np.empty(X_test.shape[0])
    for i, test_row in enumerate(X_test):
        indexes = np.argsort([-(RBF(test_row, x)) for x in X_train])[:k]
        knn = np.array([y_train[i] for i in indexes])

        y_pred[i] = count_class(knn)

    return y_pred


# In[8]:


# y_pred = predict(X_test, X_train, y_train)
# y_pred = y_pred.astype(int)


# In[9]:


# acc = np.mean(y_pred == y_test)


# In[10]:


# acc

y_pred = [None] * 3
y_pred[0] = predict(X_test, X_train, y_train, 1)
y_pred[1] = predict(X_test, X_train, y_train, 3)
y_pred[2] = predict(X_test, X_train, y_train, 7)


# In[11]:


# correct = 0
# for i in range(y_pred.shape[0]):
#     if(y_pred[i] == y_test[i]):
#         correct += 1


# In[12]:


# acc
y_pred[0] = y_pred[0].astype(int)
y_pred[1] = y_pred[1].astype(int)
y_pred[2] = y_pred[2].astype(int)


# In[13]:


acc0 = np.mean(y_pred[0] == y_test)
acc1 = np.mean(y_pred[1] == y_test)
acc2 = np.mean(y_pred[2] == y_test)


# In[14]:


print(acc0, ", ",  acc1, ", ", acc2)

