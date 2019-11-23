
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import math

from scipy.stats import norm
from scipy.stats import laplace
import statistics as stat
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
df = pd.read_csv("../spambase/spambase.data", names = word_labels, header=None) 
# df_norm = df.iloc[:, :-1]
# df_norm = (df_norm - df_norm.mean()) / df_norm.std()
# df = df_norm.join(df.iloc[:, -1])


# In[ ]:





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


df.iloc[1, :]


# In[5]:


# test_df.head()


# In[6]:


# from random import randrange
# def make_kfolds(df, kfolds):
#     data_folds = list()
#     df_list = df.values.tolist()
#     fold_size = int(len(df) / kfolds)
#     for i in range(kfolds):
#         fold = list()
#         while len(fold) < fold_size:
#             x = len(df_list);
#             index = randrange(x)
#             fold.append(df_list.pop(index))
#         data_folds.append(pd.DataFrame(fold))
#     return data_folds

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


# In[7]:


data_folds = make_kfolds(df, 10)


# In[8]:


data_folds[1].head()


# In[9]:


# X =train_df.iloc[:, :-1]
# y =train_df.iloc[:, -1]


# In[10]:


# len(labels)


# In[11]:


#labels = ['min_value', 'low-mean-value', 'overall-mean-value', 'high-mean-value', 'max-value']
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
for j in df.columns[:-1]:
    mean = df[j].mean()
    mean1_df = df.loc[df['label'] == 1][j].mean()
    mean2_df = df.loc[df['label'] == 0][j].mean()
    if(mean1_df < mean2_df):
        lo = mean1_df 
        hi = mean2_df
    else:
        lo = mean2_df
        hi = mean1_df
    bins = [-1*np.inf, lo/2, lo, (lo+mean)/2, mean, (mean+hi)/2, hi, hi*2, hi*4, np.inf]
    df[j] = df[j].replace(0,mean) 
    df[j] = pd.cut(df[j],bins,labels=labels)


# In[12]:


def count(data,feature_name,label,target):
    cond = (data[feature_name] == label) & (data['label'] == target)
    return len(data[cond])


# In[13]:


prob_dict = {0:{},1:{}}


# In[14]:


random.seed(0)
train, test = train_test_split(df, 0.20)

train_X = train
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]


# In[15]:


cnt_label_0 = count(train_X,'label',0,0)
cnt_label_1 = count(train_X,'label',1,1)
    
probability0 = cnt_label_0/len(train_X)
probability1 = cnt_label_1/len(train_X)


# In[16]:


cnt_label_1


# In[17]:


for col in train_X.columns[:-1]:
        prob_dict[0][col] = {}
        prob_dict[1][col] = {}
        
        for feature_cat in labels:
            cnt_cat_0 = count(train_X,col,feature_cat,0)
            cnt_cat_1 = count(train_X,col,feature_cat,1)
            
            prob_dict[0][col][feature_cat] = (cnt_cat_0 +1) / (cnt_label_0+9)
            prob_dict[1][col][feature_cat] = (cnt_cat_1+1) / (cnt_label_1+9)


# In[18]:


prob_dict


# In[19]:


y_pred = []


# In[20]:


for i in range(0,len(test_X)):
        product0 = probability0
        product1 = probability1
        for column in test_X.columns:   #p of x give y
            product0 *= prob_dict[0][column][test_X[column].iloc[i]]
            product1 *= prob_dict[1][column][test_X[column].iloc[i]]
        
        #Predict the outcome
        if product0 > product1:
            y_pred.append(0)
        else:
            y_pred.append(1)


# In[21]:


tp,tn,fp,fn = 0,0,0,0
for j in range(0,len(y_pred)):
    if y_pred[j] == 0:
        if test_y.iloc[j] == 0:
            tp += 1
        else:
            fp += 1
    else:
        if test_y.iloc[j] == 1:
            tn += 1
        else:
            fn += 1


# In[22]:


accuracy = (tp + tn)/len(test_y)


# In[23]:


print('Accuracy for training length ',((tp+tn)/len(test_y))*100)


# In[24]:


thresholds = np.linspace(2,-2,105)

ROC = np.zeros((105,2))

for i in range(105):
    t = thresholds[i]

    TP_t = np.logical_and( y_pred > t, test_y==1 ).sum()
    TN_t = np.logical_and( y_pred <=t, test_y==0 ).sum()
    FP_t = np.logical_and( y_pred > t, test_y==0 ).sum()
    FN_t = np.logical_and( y_pred <=t, test_y==1 ).sum()

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


# In[25]:


AUC = 0.
for i in range(100):
    AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
AUC *= 0.5
AUC

