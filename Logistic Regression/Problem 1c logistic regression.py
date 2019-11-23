
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.optimize as sp
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


df_norm = df.iloc[:, :-1]
df_norm = (df_norm - df_norm.mean()) / (df_norm.max() - df_norm.min())
df = df_norm.join(df.iloc[:, -1])
df


# In[4]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[5]:


random.seed(1)
train_df, test_df = train_test_split(df, 0.2)
train_df.head()


# In[6]:


def sigmoid(theta, X): 

    return 1.0/(1 + np.exp(-np.dot(X, theta.T))) 


# In[7]:


def gradient(theta, X, y): 

    hypothesis_cost = sigmoid(theta, X) - y.reshape(X.shape[0], -1) 
    gradient = np.dot(hypothesis_cost.T, X) 
    return gradient 
  


# In[8]:


def cost_function(theta, X, y): 

    hypothesis = sigmoid(theta, X) 
    y = np.squeeze(y) 
    term1 = y * np.log(hypothesis) 
    term2 = (1 - y) * np.log(1 - hypothesis) 
    calculate_cost = -term1 - term2 
    return np.mean(calculate_cost) 
  


# In[9]:


def gradient_descent(X, y, theta, alpha=.001, max_cost=.001): 

    cost = cost_function(theta, X, y) 
    change_cost = np.inf
      
    while(change_cost > max_cost): 
        old_cost = cost 
        theta = theta - (alpha * gradient(theta, X, y)) 
        cost = cost_function(theta, X, y) 
        change_cost = old_cost - cost 
      
    return theta
  


# In[10]:


def predict_values(theta, X): 

    pred_prob = sigmoid(theta, X) 
    pred_value = np.where(pred_prob >= .5, 1, 0) 
    return np.squeeze(pred_value) 


# In[11]:



X = train_df.iloc[:, :-1]
X = np.array(X)


X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X)) 

 
y = train_df.iloc[:, -1] 
y = np.array(y)

theta = np.matrix(np.zeros(X.shape[1])) 


theta = gradient_descent(X, y, theta) 


y_pred = predict_values(theta, X) 

print("Accuracy:", np.mean(y == y_pred)) 


# In[12]:


X_test = test_df.iloc[:, :-1]
X_test = np.array(X_test)


X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test)) 


y_test = test_df.iloc[:, -1] 
y_test = np.array(y_test)


# In[13]:


y_pred_test = predict_values(theta, X_test) 


# In[14]:


print("Accuracy:", np.mean(y_test == y_pred_test)) 


# In[15]:


thresholds = np.linspace(2,-2,105)

ROC = np.zeros((105,2))

for i in range(105):
    t = thresholds[i]

    TP_t = np.logical_and( y_pred_test > t, y_test==1 ).sum()
    TN_t = np.logical_and( y_pred_test <=t, y_test==0 ).sum()
    FP_t = np.logical_and( y_pred_test > t, y_test==0 ).sum()
    FN_t = np.logical_and( y_pred_test <=t, y_test==1 ).sum()

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


# In[16]:


AUC = 0.
for i in range(100):
    AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
AUC *= 0.5
AUC

