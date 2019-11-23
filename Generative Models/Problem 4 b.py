
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as sp
from random import random as rand


# In[2]:


df = pd.read_csv("3gaussian.txt", delim_whitespace=True,  header=None) 


# In[3]:


df.head()


# In[4]:


X = np.array(df)


# In[5]:


m, n = X.shape
data = X.copy()

k = 3

mean = np.asmatrix(np.random.random((k, n)))
sigma = np.array([np.asmatrix(np.identity(n)) for i in range(k)])
phi = np.ones(k)/k
w = np.asmatrix(np.empty((m, k), dtype=float))


# In[6]:


tolerance=1e-4
iteratins = 0
llh = 1
previous_llh = 0
while(llh-previous_llh > tolerance):
    previous_llh = llh_function()
    e_step()
    m_step()
    iteratins += 1
    llh = llh_function()
    print('loglikelihood: ', llh)


# In[7]:


def llh_function():
    llh = 0
    for i in range(m):
        p = 0
        for j in range(k):
            #print(self.sigma[j])
            p += sp.multivariate_normal.pdf(data[i, :], mean[j, :].A1, sigma[j, :]) 
        llh += np.log(p) 
    return llh
    


# In[9]:


def e_step():
    for i in range(m):
        sum = 0
        for j in range(k):
            q = sp.multivariate_normal.pdf(data[i, :], mean[j].A1, sigma[j]) 
            sum += q
            wt[i, j] = q
        wt[i, :] /= sum
        assert wt[i, :].sum() - 1 < 1e-4

 


# In[10]:


def m_step():
    for j in range(k):
        const = w[:, j].sum()
        phi[j] = 1/m * const
        muj = np.zeros(n)
        sigmaj = np.zeros((n, n))
        for i in range(m):
            muj += (data[i, :] * w[i, j])
            sigmaj += w[i, j] * ((data[i, :] - mean[j, :]).T * (data[i, :] - mean[j, :]))
        mean[j] = muj / const
        sigma[j] = sigmaj / const


# In[12]:


mean


# In[13]:


sigma

