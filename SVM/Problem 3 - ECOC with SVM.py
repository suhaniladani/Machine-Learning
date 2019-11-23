
# coding: utf-8

# In[1]:


from __future__ import division, print_function
import os
import numpy as np
import random as rnd
import pandas as pd
import random
from sklearn.decomposition import PCA


# In[2]:


arr_train = np.load("../haar_train_full.npy")
arr_test = np.load("../haar_test_full.npy")

# df_X = pd.read_csv("../Haar_Features/X_df.csv") 
# df_y = pd.read_csv("../Haar_Features/y_df.csv") 
# df_X = (df_X - df_X.mean()) / df_X.std()


# In[3]:


# X = np.array(df_X)
# y = np.array(df_y)
# y = np.squeeze(y)


train_df = pd.DataFrame(arr_train)
test_df = pd.DataFrame(arr_test)
test_df.head()
df = train_df.append(test_df)
df_norm = df.iloc[:, :-1]
df_norm = (df_norm - df_norm.mean()) / df_norm.std()
df = df_norm.join(df.iloc[:, -1])

# train_feature = train_df.iloc[:, :-1]
# test_feature = test_df.iloc[:, :-1]

# WholeDf = pd.concat([train_feature, test_feature])


# In[4]:


train_df = df.iloc[0:60000, :]
test_df = df.iloc[60000: , :]


# In[5]:


train_df.shape


# In[6]:


# def train_test_split(df, test_size):
    
#     if isinstance(test_size, float):
#         test_size = round(test_size * len(df))

#     index_list = df.index.tolist()
#     test_indices = random.sample(population=index_list, k=test_size)

#     test_df = df.loc[test_indices]
#     train_df = df.drop(test_indices)
    
#     return train_df, test_df


# In[7]:


X_train = np.array(train_df.iloc[: , :-1])
y_train = np.array(train_df.iloc[: , -1])

X_test = np.array(test_df.iloc[: , :-1])
y_test = np.array(test_df.iloc[: , -1])


# In[8]:


# train_features = principalDf.iloc[0:12000]
# test_features = principalDf.iloc[60000:]


# In[9]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
uni_code = {}


# In[10]:


count = 0
while (count!= 10):
    uni_code[count] = np.random.randint(2, size=50)
    count = count+1


# In[11]:


y_test_mat = [None] * len(y_test)
y_train_mat = [None] * len(y_train)


# In[12]:


# random.seed(3)
# train_df, test_df = train_test_split(df, 0.20)
for i in range(len(y_test)):
    y_test_mat[i] = uni_code.get(int(y_test[i]))
    
for i in range(len(y_train)):
    y_train_mat[i] = uni_code.get(int(y_train[i]))


# In[13]:


y_test_df = pd.DataFrame.from_records(y_test_mat)
X_test_df = pd.DataFrame.from_records(X_test)

y_train_df = pd.DataFrame.from_records(y_train_mat)
X_train_df = pd.DataFrame.from_records(X_train)


# In[14]:


y_train_df = y_train_df.replace(0, -1)
y_test_df = y_test_df.replace(0, -1)
y_train_df.head()


# In[15]:


y_train_mat


# In[16]:


y_train[0]


# In[17]:


max_iter=200
C=0.01
epsilon=0.001


# In[18]:


def fit(X, y):
    n, d = X.shape[0], X.shape[1]
    alpha = np.zeros((n))
    
    kernel = kernel_linear
    count = 0
    kernel_type = 'linear'
    while True:
        count += 1
        alpha_previous = np.copy(alpha)
        for j in range(0, n):
            i = j
            cnt=0
            while i == j and cnt<1000:
                i = rnd.randint(0, n-1)
                cnt=cnt+1
            xi, xj, yi, yj = X[i,:], X[j,:], y[i], y[j]
            k_func_ij = kernel(xi, xi) + kernel(xj, xj) - 2 * kernel(xi, xj)
            if k_func_ij == 0:
                continue
            alpha_j, alpha_i = alpha[j], alpha[i]
            (L, H) = calc_L_H(C, alpha_j, alpha_i, yj, yi)

            w = calculate_w(alpha, y, X)
            b = calculate_b(X, y, w)

            Ei = prediction_func(xi, w, b) - yi
            Ej = prediction_func(xj, w, b) - yj

            alpha[j] = alpha_j + float(yj * (Ei - Ej))/k_func_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)

            alpha[i] = alpha_i + yi*yj * (alpha_j - alpha[j])

        difference = np.linalg.norm(alpha - alpha_previous)
        if difference < epsilon:
            break

        if count >= max_iter:
            print("increase num of iterations" % (max_iter))
            return

    b = calculate_b(X, y, w)
    if kernel_type == 'linear':
        w = calculate_w(alpha, y, X)

    alpha_index = np.where(alpha > 0)[0]
    sv = X[alpha_index, :]
    return sv, count, w, b


# In[19]:


def predict(X):
    return prediction_func(X, w, b)


# In[20]:


def calculate_b(X, y, w):
    b_tmp = y - np.dot(w.T, X.T)
    return np.mean(b_tmp)


# In[21]:


def calculate_w(alpha, y, X):
    return np.dot(X.T, np.multiply(alpha,y))


# In[22]:


def prediction_func(X, w, b):
    return np.sign(np.dot(w.T, X.T) + b).astype(int)


# In[24]:


def calc_L_H(C, alpha_prime_j, alpha_prime_i, y_j, y_i):
    if(y_i != y_j):
        return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
    else:
        return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))


# In[26]:


def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)


# In[27]:


def kernel_quadratic(x1, x2):
    return (np.dot(x1, x2.T) ** 2)


# In[29]:


# y_test = y_test.replace(0, -1)


# In[30]:


# y_test = np.array(y_test)


# In[31]:


# y_test


# In[32]:


# np.mean(y_test == y_pred)


# In[ ]:


y_pred_mat = []
for i in range (50):
#     print(y_train_df[i])
    support_vectors, count, w, b = fit(X_train, np.array(y_train_df[i]))
    y_pred = predict(X_test)
    print(y_pred)
    y_pred_mat.append(y_pred)


# In[ ]:


np.array(y_train_df[0])


# In[ ]:


y_pred_trans = np.array(y_pred_mat).transpose()


# In[ ]:


y_pred_trans


# In[ ]:


y_pred_trans[y_pred_trans == -1] = 0


# In[ ]:


import csv


with open("digit_y_pred.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(y_pred_trans)


# In[ ]:


def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


# In[ ]:


y_pred_coded = np.loadtxt(open("digit_y_pred.csv", "rb"), delimiter=",")


# In[ ]:


y_pred_coded.shape
y_test = np.squeeze(y_test)


# In[ ]:


y_pred_coded = y_pred_coded.astype(int)
x = np.array(y_test).shape[0]


# In[ ]:


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


# In[ ]:


y_test


# In[ ]:


np.array(y_pred_decoded)


# In[ ]:


acc = np.mean(y_test == np.array(y_pred_decoded))


# In[ ]:


acc

