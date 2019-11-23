
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
df_norm = df.iloc[:, :-1]
df_norm = (df_norm - df_norm.mean()) / df_norm.std()
df = df_norm.join(df.iloc[:, -1])


# In[3]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indices = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[4]:


random.seed(0)
train_df, test_df = train_test_split(df, test_size=0.20)


# In[5]:


theta_len = len(df.columns)
theta_len


# In[6]:



x = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]


x = np.c_[np.ones(x.shape[0]), x] 
x


# In[7]:


alpha = 0.01 
iterations = 2000 
m = y.size 
np.random.seed(0) 
theta = np.random.rand(theta_len) 


def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        hypothesis = np.dot(x, theta)
        error = hypothesis - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs


past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)
theta = past_thetas[-1]


# In[8]:


theta


# In[9]:


y_pred = theta.T.dot(x.T)
y_pred


# In[10]:


y_pred = y_pred.T 
mse = np.mean((y - y_pred)**2)
mse


# In[11]:


threshold = 0.4
y_pred[y_pred < threshold] = 0
y_pred[y_pred > threshold] = 1


# In[12]:


print("Accuracy:", np.mean(y == y_pred)) 


# In[13]:


costs = past_costs[-1]
costs


# In[14]:


#Grab the relevant data, scale the predictor variable, and add a column of 1s for the gradient descent...
x_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]


x_test = np.c_[np.ones(x_test.shape[0]), x_test] 
x_test


# In[15]:


y_pred_test = theta.T.dot(x_test.T)
y_pred_test


# In[16]:


y_pred_test = y_pred_test.T 
mse_test = np.mean((y_test - y_pred_test)**2)
mse_test


# In[17]:


y_pred_test_df = pd.DataFrame(y_pred_test)
y_test.shape


# In[18]:


TP = np.sum(np.logical_and(y_pred_test == 1, y_test == 1))
 
TN = np.sum(np.logical_and(y_pred_test == 0, y_test == 0))
 
FP = np.sum(np.logical_and(y_pred_test == 1, y_test == 0))
 
FN = np.sum(np.logical_and(y_pred_test == 0, y_test == 1))


# In[19]:


thresholds = np.linspace(1,0,105)

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
y_pred_test[y_pred_test < threshold] = 0
y_pred_test[y_pred_test > threshold] = 1


# In[22]:


print("Accuracy:", np.mean(y_test == y_pred_test)) 

