
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import scipy.stats as stat
from random import random as rand


# In[15]:


df = pd.read_csv("2gaussian.txt", delim_whitespace=True,  header=None) 


# In[16]:


df.head()


# In[17]:


X = np.array(df)


# In[18]:


m, n = X.shape
data = X.copy()

k = 2

mean = np.asmatrix(np.random.random((k, n)))
sigma = np.array([np.asmatrix(np.identity(n)) for i in range(k)])
phi = np.ones(k)/k
wt = np.asmatrix(np.empty((m, k), dtype=float))
    


# In[19]:


tolerance=1e-4
iterations = 0
llh = 1
previous_llh = 0
while(llh-previous_llh > tolerance):
    previous_llh = llh_function()
    e_step()
    m_step()
    iterations += 1
    llh = llh_function()
    print('loglikelihood: ', llh)
    


# In[20]:


def llh_function():
    llh = 0
    for i in range(m):
        p = 0
        for j in range(k):
            #print(self.sigma[j])
            p += sp.multivariate_normal.pdf(data[i, :],  mean[j, :].A1, sigma[j, :])
        llh += np.log(p) 
    return llh
    


# In[22]:


def e_step():
    for i in range(m):
        sum = 0
        for j in range(k):
            q = stat.multivariate_normal.pdf(data[i, :], mean[j].A1, sigma[j])   
            sum += q
            wt[i, j] = q
        wt[i, :] /= sum
        assert wt[i, :].sum() - 1 < 1e-4


# In[23]:


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


# In[25]:


mean


# In[26]:


sigma

