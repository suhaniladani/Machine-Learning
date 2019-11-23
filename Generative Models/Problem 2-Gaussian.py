
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import random


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


df.head()


# In[4]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indexes = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    
    return train_df, test_df


# In[5]:


random.seed(3)
train_df, test_df = train_test_split(df, 0.20)


# In[7]:


from random import randrange
def make_kfolds(df, kfolds):
    data_folds = list()
    for i in range(kfolds):
        data_folds.append(pd.DataFrame())
    counter = 0
    for i in range(len(df)):
        if counter >= kfolds:
            counter = 0
        data_folds[counter] = data_folds[counter].append(df[i:i+1])
        counter += 1
    return data_folds


# In[8]:


data_folds = make_kfolds(df, 10)
data_folds[1].head()


# In[9]:


X_train = train_df.iloc[: , :-1]
y_train = train_df.iloc[: , -1]

X_test = test_df.iloc[: , :-1]
y_test = test_df.iloc[: , -1]


# In[10]:


num_spam = df['label'][df['label'] == 1].count()
num_non_spame = df['label'][df['label'] == 0].count()
total = len(df)

print('Spam:',num_spam)
print('Non-spam ',num_non_spame)
print('Total: ',total)


# In[11]:


prob_spam = num_spam/total
print('Probability spam: ',prob_spam)

prob_non_spam = num_non_spame/total
print('Probability non-spam: ',prob_non_spam)


# In[12]:



data_mean = df.groupby('label').mean()

data_variance = df.groupby('label').var()*(1/6)


# In[13]:


def prob_x_y(x, mean_y, variance_y):
    prob = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return prob


# In[14]:


y_pred = []


# In[15]:


for row in range(0,len(X_train)):
        prod_0 = prob_non_spam
        prod_1 = prob_spam
        for col in X_train.columns:   
            prod_0 *= prob_x_y(X_train[col].iloc[row], data_mean[col][0], data_variance[col][0])
            prod_1 *= prob_x_y(X_train[col].iloc[row], data_mean[col][1], data_variance[col][1])
    

        if prod_0 > prod_1:
            y_pred.append(0)
        else:
            y_pred.append(1)


# In[16]:



np.array(y_train)


# In[17]:


np.mean(np.array(y_pred) == np.array(y_train))


# In[18]:


thresholds = np.linspace(2,-2,105)

ROC = np.zeros((105,2))

for i in range(105):
    t = thresholds[i]
    
    TP_t = np.logical_and( y_pred > t, y_train==1 ).sum()
    TN_t = np.logical_and( y_pred <=t, y_train==0 ).sum()
    FP_t = np.logical_and( y_pred > t, y_train==0 ).sum()
    FN_t = np.logical_and( y_pred <=t, y_train==1 ).sum()

    FPR_t = FP_t / float(FP_t + TN_t)
    ROC[i,0] = FPR_t

    TPR_t = TP_t / float(TP_t + FN_t)
    ROC[i,1] = TPR_t

fig = plt.figure(figsize=(6,6))
plt.plot(ROC[:,0], ROC[:,1], lw=2)
plt.xlabel('$FPR(t)$')
plt.ylabel('$TPR(t)$')
plt.grid()


# In[19]:


AUC = 0.
for i in range(100):
    AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
AUC *= 0.5
AUC

