
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
import pandas as pd
import random
from __future__ import division, print_function
import math
import gc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("../spambase/spambase.data", names = word_labels, header=None) 
df.head()


# In[3]:


# df['label'].replace(0, -1,inplace=True)
# df.head()


# In[4]:


y_test_pred = pd.read_csv("y_test_predict.csv", names = word_labels, header=None) 
y_train_pred = pd.read_csv("y_train_predict.csv", names = word_labels, header=None) 


# In[5]:


# def train_test_split(df, test_size):
    
#     if isinstance(test_size, float):
#         test_size = round(test_size * len(df))

#     index_list = df.index.tolist()
#     test_indexes = random.sample(population=index_list, k=test_size)

#     test_df = df.loc[test_indexes]
#     train_df = df.drop(test_indexes)
    
#     return train_df, test_df


# In[6]:


# random.seed(3)
# train_df, test_df = train_test_split(df, 0.20)


# In[7]:


def margin_calc(vector):
    res = 0
    x = np.sort(-vector)
    res =  x[0] - x[1]
    return res


# In[8]:


# X_train = train_df.iloc[: , :-1]
# y_train = train_df.iloc[: , -1]

# X_test = test_df.iloc[: , :-1]
# y_test = test_df.iloc[: , -1]
X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2014)


# In[9]:


X_train


# In[10]:


abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)

model = abc.fit(X_train, y_train)


y_pred = model.predict(X_test)

prob = model.predict_proba(X_test)


# In[11]:


def active_learning(clf, X, y, test_x, test_y, p_p_s, initial_size):


    train_ids = np.array(range(initial_size))
    acc = []
    train_size = []
    train_list = []
    mean_margin = []

    group_inds = np.array(range(0, len(y)))

    
  
    while len(group_inds) > p_p_s:

        train_y = y[train_ids]
        train_x = X[train_ids,:]
        group_x = np.delete(X, train_ids, 0)
        group_inds = np.delete(np.array(range(0, len(y))), train_ids)
    
        train_size.append(len(train_ids))
        train_list.append(train_ids)

    
        model = clf
        model.fit(train_x, train_y)


        try:
            probs = model.predict_proba(group_x)
        except:
            temp_proba = model.decision_function(group_x)
            temp_proba = 1.0/(1.0+np.exp(-temp_proba))
            if len(temp_proba.shape)==1:
                probs = np.zeros([len(temp_proba), 2])
                probs[:,0] = temp_proba
                probs[:,1] = 1 - temp_proba
            else:
                probs = temp_proba

        unsure_data = []
        for i in probs:
            unsure_data.append(margin_calc(i))
        result = np.zeros([len(group_inds), 2])
        result[:, 0] = group_inds
        result[:, 1] = unsure_data
        scores = np.array(sorted(result, key=lambda w: -w[1]))

        pred = model.predict(test_x)
        accuracy = np.mean(pred == test_y)
        acc.append(accuracy)
        
        mean_margin.append(np.mean(scores[0: p_p_s, 1]))

        train_ids = np.concatenate([train_ids, np.array(scores[0: p_p_s, 0]).astype(int)])

        del train_x
        del train_y
        del group_x
        del model
        
    return (train_size, acc, train_list, clf, mean_margin)


# In[12]:


clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)

active_learn_result = active_learning(clf, X_train, y_train, X_test, y_test, 500, 500)



plt.figure(figsize=(11, 9))
plt.plot(active_learn_result[0], active_learn_result[1])
plt.xlabel("Train size")
plt.ylabel("Accuracy")
plt.grid()


# In[13]:


active_learn_result[1]

