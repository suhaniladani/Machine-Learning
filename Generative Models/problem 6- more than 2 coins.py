
# coding: utf-8

# In[1]:


import random
import numpy as np
import operator


# In[3]:


np.random.seed(2)
def flip(p):
    return 'H' if np.random.random() < p else 'T'


# In[4]:


def generate_mixture(p):
    return flip(0.75) if np.random.random() < p else flip(0.5)


# In[5]:


flips = list()
for i in range(1000):
    flips.append([generate_mixture(0.8) for i in range(10)])


# In[6]:


flips
T = 2


# In[7]:


# n= 10
# p = 0.75
# r = 0.4
# s = np.random.binomial(n, p, 800)
# t = np.random.binomial(n, r, 200)
# flips = list()
# flips.append(s)
# flips.append(t)


# In[8]:


def coin_em(flips, maxiter, T):
    # Initial Guess
    pi = list()
    px = list()
    for i in range(T):
        pi.append(np.random.random())
    thetas = []
    # Iterate
    for c in range(maxiter):
        print(c, pi)
        heads, tails = e_step(flips, pi)
        pi = m_step(heads, tails, pi)
        
    thetas.append((pi))  
    return thetas


# In[9]:


def e_step(flips, pi):
   heads = list()
   tails = list()
   llh = list()
   all_p = list()
   for i in range(len(pi)):
       heads.append(0)
       tails.append(0)
       llh.append(0)
       all_p.append(0)
   for trial in flips:
       for i in range(len(pi)):
           llh[i] = coin_llh(trial, pi[i])
       for i in range(len(pi)):
           all_p[i]= llh[i] / sum(llh)
       for i in range(len(pi)):
           tails[i] += all_p[i] * trial.count("T")
           heads[i] += all_p[i] * trial.count("H")
   return heads, tails


# In[10]:


def m_step(heads, tails, pi):
    for i in range((len(pi))):
        pi[i] = (heads[i] / (heads[i] + tails[i]))
    return pi



# In[11]:


def coin_llh(flip, bias):
    # P(X | Z, p)
    numHeads = flip.count('H')
    flips_num = len(flip)
    return pow(bias, numHeads) * pow(1-bias, flips_num-numHeads)


# In[12]:



thetas = coin_em(flips, 10, T)


# In[13]:


thetas

