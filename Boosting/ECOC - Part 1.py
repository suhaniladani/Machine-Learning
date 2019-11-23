
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import re
import random
import math


# In[2]:


uni_code = {}


# In[3]:


count = 0
while (count!= 8):
    uni_code[count] = np.random.randint(2, size=20)
    count = count+1


# In[4]:


uni_code


# In[5]:


# with open("8newsgroup/train.trec/feature_matrix.txt", 'r') as file_object:
#     line = file_object.readline()
#     i = 0
#     while line:
#         i = i+1
#     print(i)
        


# In[6]:


lines_train = open('8newsgroup/train.trec/feature_matrix.txt').readlines()
lines_test = open('8newsgroup/test.trec/feature_matrix.txt').readlines()


# In[7]:



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

        

                
                


# In[8]:


# y_train = pd.DataFrame(label)


# In[9]:


X_test, y_test = parse_data(lines_test, 7532, 1754)
X_train, y_train = parse_data(lines_train, 11314, 1754)


# In[10]:


y_test_mat = [None] * len(y_test)
y_train_mat = [None] * len(y_train)


# In[11]:


for i in range(len(y_test)):
    y_test_mat[i] = uni_code.get(int(y_test[i]))
    


# In[12]:


for i in range(len(y_train)):
    y_train_mat[i] = uni_code.get(int(y_train[i]))


# In[13]:


y_test_df = pd.DataFrame.from_records(y_test_mat)
X_test_df = pd.DataFrame.from_records(X_test)


# In[14]:


y_train_df = pd.DataFrame.from_records(y_train_mat)
X_train_df = pd.DataFrame.from_records(X_train)


# In[15]:


y_train_df = y_train_df.replace(0, -1)
y_test_df = y_test_df.replace(0, -1)
y_train_df.head()


# In[16]:


def make_clf_dict():
    clf_dict = {'label': 1, 'feature_name': None, 'threshold': None, 'alpha': None}
    return clf_dict


# In[17]:


def predict(X):
    n_samples = np.shape(X)[0]
    y_pred = np.zeros((n_samples, 1))
    for clf in classifiers:
        predictions = np.ones(np.shape(y_pred))
        index_neg = (clf['label'] * X[clf['feature_name']] < clf['label'] * clf['threshold'])
        predictions[index_neg] = -1
        y_pred += clf['alpha'] * predictions
    y_pred = np.sign(y_pred).flatten()

    return y_pred



# In[18]:


iters = 200

def adaboost(X, y):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    classifiers = []
    for i in range(iters):
        clf = make_clf_dict()
        min_err = float('inf')
        for column in X_train_df.columns:
            feature_values = np.expand_dims(X[column], axis=1)
            unique_values = np.unique(feature_values)
 
            for threshold in unique_values:
                p = 1
    
                prediction = np.ones(np.shape(y))
             
                prediction[X[column] < threshold] = -1

                error = sum(w[y != prediction])

                if error > 0.5:
                    error = 1 - error
                    p = -1

                if error < min_err:
                    clf['label'] = p
                    clf['threshold'] = threshold
                    clf['feature_name'] = column
                    min_err = error

        clf['alpha'] = 0.5 * math.log((1.0 - min_err) / (min_err + 1e-10))

        predictions = np.ones(np.shape(y))

        index_neg = (clf['label'] * X[clf['feature_name']] < clf['label'] * clf['threshold'])

        predictions[index_neg] = -1

        w *= np.exp(-clf['alpha'] * y * predictions)

        w /= np.sum(w)

        classifiers.append(clf)
    return classifiers


# In[19]:


y_pred_mat = []
for i in range (8):
    classifiers = adaboost(X_train_df, y_train_df[i])
    y_pred = predict(X_test_df)
    print(y_pred)
    y_pred_mat.append(y_pred)


# In[20]:


y_pred


# In[21]:


y_pred_mat


# In[22]:


y_pred_trans = np.array(y_pred_mat).transpose()


# In[23]:


p = y_train_df[i].to_frame()


# In[25]:


classifiers


# In[27]:


for i in range (8, 20):
    classifiers = adaboost(X_train_df, y_train_df[i])
    y_pred = predict(X_test_df)
    print(y_pred)
    y_pred_mat.append(y_pred)


# In[28]:


y_pred_trans = np.array(y_pred_mat).transpose()


# In[51]:


y_pred_trans[y_pred_trans == -1] = 0


# In[39]:


np.array(y_test_df).shape


# In[38]:


# import xlsxwriter

# workbook = xlsxwriter.Workbook('arrays.xlsx')
# worksheet = workbook.add_worksheet()


# row = 0

# for col, data in enumerate(y_pred_mat):
#     worksheet.write_column(row, col, data)

# workbook.close()


# In[77]:


# def hamming2(s1, s2):
#     """Calculate the Hamming distance between two bit strings"""
#     assert len(s1) == len(s2)
#     return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero( (a and r) != (b and r) )


# In[43]:


x = np.array(y_test_df).shape[0]


# In[44]:


y_test_matrix = np.array(y_test_df)


# In[69]:


y_pred_trans = y_pred_trans.astype(int)


# In[83]:


# import csv


# with open("new_file.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(y_pred_trans)


# In[84]:


import csv


with open("new_file2.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(y_pred_mat)

