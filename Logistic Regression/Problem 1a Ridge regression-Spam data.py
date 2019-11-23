
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
df.shape


# In[3]:


df_norm = df.iloc[:, :-1]
df_norm = (df_norm - df_norm.mean()) / (df_norm.max() - df_norm.min())
df = df_norm.join(df.iloc[:, -1])
df


# In[4]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indices = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[5]:


random.seed(0)
train_df, test_df = train_test_split(df, test_size=0.20)


# In[7]:


col_nums = train_df.shape[1]

train_X = train_df.iloc[:,0:col_nums-1]  
train_y = train_df.iloc[:,col_nums-1:col_nums] 


# In[8]:


X = np.array(train_X)
ones = np.ones(len(X))
X = np.column_stack((ones,X))
y = np.array(train_y)   


# In[9]:


def ridge_regression(X, y, lam):
    

    Xt = np.transpose(X)
    lambda_identity = lam*np.identity(len(Xt))
    inverse_term = np.linalg.inv(np.dot(Xt, X)+lambda_identity)
    w = np.dot(np.dot(inverse_term, Xt), y)
    return w


# In[10]:


# test_df.insert(0, 'Ones', 1)
# col_nums = test_df.shape[1]
# X = test_df.iloc[:,0:col_nums-1]  
# y = test_df.iloc[:,col_nums-1:col_nums] 
# y_pred = coefficient.T.dot(X.T)

# y_pred


# In[11]:


# y_pred = y_pred.T 
# mse = np.mean((y - y_pred)**2)
# mse


# In[12]:


w = ridge_regression(X, y, 0.001)


# In[13]:


w


# In[14]:


y_pred_train = w.T.dot(X.T)
y_pred_train = y_pred_train.T 
mse_train = np.mean((y - y_pred_train)**2)
mse_train


# In[15]:


threshold = 0.4
y_pred_train[y_pred_train < threshold] = 0
y_pred_train[y_pred_train > threshold] = 1


# In[16]:


print("Accuracy:", np.mean(y == y_pred_train)) 


# In[17]:



col_nums = test_df.shape[1]
X_test = test_df.iloc[:,0:col_nums-1]  
y_test = test_df.iloc[:,col_nums-1:col_nums] 

X_test = np.array(X_test)
ones = np.ones(len(X_test))
X_test = np.column_stack((ones,X_test))
y_test = np.array(y_test)

y_pred = w.T.dot(X_test.T)


# In[18]:


y_pred = y_pred.T 
mse = np.mean((y_test - y_pred)**2)
mse


# In[19]:


thresholds = np.linspace(1,0,105)

ROC = np.zeros((105,2))

for i in range(105):
    t = thresholds[i]

    TP_t = np.logical_and( y_pred > t, y_test==1 ).sum()
    TN_t = np.logical_and( y_pred <=t, y_test==0 ).sum()
    FP_t = np.logical_and( y_pred > t, y_test==0 ).sum()
    FN_t = np.logical_and( y_pred <=t, y_test==1 ).sum()

    
    FPR_t = FP_t / float(FP_t + TN_t)
    ROC[i,0] = FPR_t


    TPR_t = TP_t / float(TP_t + FN_t)
    ROC[i,1] = TPR_t

# Plot the ROC curve.
fig = plt.figure(figsize=(6,6))
plt.plot(ROC[:,0], ROC[:,1], lw=2)
plt.xlabel('$FPR(t)$')
plt.ylabel('$TPR(t)$')
plt.grid()


# In[20]:


AUC = 0.
for i in range(100):
    AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
AUC *= 0.5
AUC 


# In[21]:


threshold = 0.4
y_pred[y_pred < threshold] = 0
y_pred[y_pred > threshold] = 1
y_pred.shape


# In[22]:


print("Accuracy:", np.mean(y_test == y_pred)) 


# In[23]:


TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
 
TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
 
FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
 
FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))


# In[24]:


print("TP=", TP, "TN=",TN, "FP=",FP, "FN=",FN)

