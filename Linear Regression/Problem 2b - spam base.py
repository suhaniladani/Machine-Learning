
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("spambase/spambase.data", names = word_labels, header=None) 


# In[3]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[4]:


train_df, test_df = train_test_split(df, 0.2)


# In[5]:


from random import randrange
def make_kfold(df, kfolds):
    data_folds = list()
    df_list = df.values.tolist()
    fold_size = int(len(df) / kfolds)
    for i in range(kfolds):
        fold = list()
        while len(fold) < fold_size:
            x = len(df_list);
            index = randrange(x)
            fold.append(df_list.pop(index))
        data_folds.append(pd.DataFrame(fold))
    return data_folds


# In[6]:


data_folds = make_kfold(df, 5)
data_folds[0]


# In[27]:


train_df.insert(0, 'Ones', 1)
col_nums = train_df.shape[1]

X = train_df.iloc[:,0:col_nums-1]  
y = train_df.iloc[:,col_nums-1:col_nums] 


# In[ ]:


def make_kfold_datasets(data_folds):
    train_sets = list()
    test_sets = list()
    for i in range(len(data_folds)):
        df_new = pd.DataFrame()
        for j in range(len(data_folds)):
            if(j != i):
                df_new = df_new.append(data_folds[j])
                df_new = create_labels(df_new)
                print(df_new)
            else:
                df_new_test = pd.DataFrame()
                df_new_test = create_labels(data_folds[j])
                tests.append(df_new_test)
        tree = decision_tree_algorithm(df_new)
        trees.append(tree)
    return trees, tests


# In[28]:


IdentitySize = X.shape[1]
IdentityMatrix= np.zeros((IdentitySize, IdentitySize))
np.fill_diagonal(IdentityMatrix, 1)


# In[29]:


a_const = 1
xTx = X.T.dot(X) + a_const * IdentityMatrix
XtX = np.linalg.pinv(xTx)
XtX_xT = XtX.dot(X.T)
coefficients = XtX_xT.dot(y)


# In[30]:


coefficients


# In[31]:


test_df.insert(0, 'Ones', 1)
col_nums = test_df.shape[1]
X = test_df.iloc[:,0:col_nums-1]  
y = test_df.iloc[:,col_nums-1:col_nums] 
y_pred = coefficients.T.dot(X.T)

y_pred


# In[32]:


y_pred = y_pred.T 
mse = np.mean((y - y_pred)**2)
mse

