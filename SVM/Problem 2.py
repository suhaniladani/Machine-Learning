
# coding: utf-8

# In[1]:


from __future__ import division, print_function
import os
import numpy as np
import random as rnd
import pandas as pd
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
# df_norm = df.iloc[:, :-1]
# df_norm = (df_norm - df_norm.mean()) / df_norm.std()
# df = df_norm.join(df.iloc[:, -1])


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


X_train = train_df.iloc[: , :-1]
y_train = train_df.iloc[: , -1]

X_test = test_df.iloc[: , :-1]
y_test = test_df.iloc[: , -1]


# In[6]:


max_iter=200
C=0.01
epsilon=0.001


# In[7]:


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


# In[8]:


def predict(X):
    return prediction_func(X, w, b)


# In[9]:


def calculate_b(X, y, w):
    temp_b = y - np.dot(w.T, X.T)
    return np.mean(temp_b)


# In[10]:


def calculate_w(alpha, y, X):
    return np.dot(X.T, np.multiply(alpha,y))


# In[11]:


def prediction_func(X, w, b):
    return np.sign(np.dot(w.T, X.T) + b).astype(int)


# In[13]:


def calc_L_H(C, alpha_j, alpha_i, yj, yi):
    if(yi != yj):
        return (max(0, alpha_j - alpha_i), min(C, C - alpha_i + alpha_j))
    else:
        return (max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j))


# In[14]:


def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)


# In[16]:


def kernel_quadratic(x1, x2):
    return (np.dot(x1, x2.T) ** 2)


# In[17]:


sv, count, w, b = fit(np.array(X_train), np.array(y_train))
y_pred = predict(np.array(X_test))


# In[18]:


y_test = y_test.replace(0, -1)


# In[19]:


y_test = np.array(y_test)


# In[20]:


y_test


# In[21]:


np.mean(y_test == y_pred)

