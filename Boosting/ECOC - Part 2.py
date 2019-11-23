
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import random
import math
import re


# In[2]:


uni_code = {}


# In[3]:


count = 0
while (count!= 8):
    uni_code[count] = np.random.randint(2, size=20)
    count = count+1


# In[4]:


lines_train = open('8newsgroup/train.trec/feature_matrix.txt').readlines()
lines_test = open('8newsgroup/test.trec/feature_matrix.txt').readlines()


# In[5]:


def parse_data(lines, n_samples, n_features):
# # DataFrame = pd.read_csv("8newsgroup/train.trec/feature_matrix.txt", delimiter=' ' , header=None) 
# # DataFrame.head()
    label = []
    data = np.empty([11314, 1754])
    # # test_data = np.empty((num_test_rows, num_features), dtype=float)
    # with open("8newsgroup/train.trec/feature_matrix.txt", 'r') as file_object:
    #     line = file_object.readline()
    #     i = 0
    for i, line in enumerate(lines, start=0):
        step_0 = re.split(' ',line)
        label.append(step_0[0])
        for v in step_0[1:-1]:
            f, c = v.split(':')
            f = int(f)
            c = float(c)
            data[i, f] = c
    return data, label

        


# In[6]:


X_test, y_test = parse_data(lines_test, 7532, 1754)
X_train, y_train = parse_data(lines_train, 11314, 1754)


# In[7]:


y_test_mat = [None] * len(y_test)
y_train_mat = [None] * len(y_train)


# In[8]:


y_pred_coded = np.loadtxt(open("new_file.csv", "rb"), delimiter=",", skiprows=1)


# In[9]:


y_pred_coded.shape


# In[10]:


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# def hamming_distance(a, b):
#     r = (1 << np.arange(8))[:,None]
#     return np.count_nonzero( (a and r) != (b and r) )


# In[11]:


x = np.array(y_test).shape[0]


# In[12]:


y_pred_coded = y_pred_coded.astype(int)


# In[13]:


y_pred_decoded = list()
for i in range(x):
    min_ham = np.inf
    code = 0
    temp = 0
    a = "".join(str(y_pred_coded[i]))
    for j in range (8):
        b = "".join(str(uni_code[j]))
#         print(a)
#         print(b)
        h = hamming_distance(a, b)
        if(h < min_ham):
            min_ham = h
            temp = j
    code = temp
    y_pred_decoded.append(code)
    


# In[14]:


counter_corr = 0
for i in range (len(y_pred_decoded)):
    if (str(y_pred_decoded[i]) == y_test[i]):
        counter_corr += 1
        


# In[15]:


len(y_pred_decoded)


# In[16]:


acc = counter_corr/7532


# In[17]:


acc

