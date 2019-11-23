
# coding: utf-8

# In[1]:


import random
import numpy as np


# In[2]:


np.random.seed(0)
def flip(p):
    return 'H' if np.random.random() < p else 'T'


# In[3]:


def generate_mixture(p):
    return flip(0.75) if np.random.random() < p else flip(0.4)


# In[4]:


flips = list()
for i in range(1000):
    flips.append([generate_mixture(0.8) for i in range(10)])


# In[5]:


flips


# In[6]:


# n= 10
# p = 0.75
# r = 0.4
# s = np.random.binomial(n, p, 800)
# t = np.random.binomial(n, r, 200)
# flips = list()
# flips.append(s)
# flips.append(t)


# In[7]:


def coin_em(flips, maxiter):
    p = 0.2
    r = 0.5
    thetas = []
    # Iterate
    for c in range(maxiter):
        print("#%d:\t%0.2f %0.2f" % (c, p, r))
        heads_coin_p, tails_coin_p, heads_coin_q, tails_coin_q = e_step(flips, p, r)
        p, r = m_step(heads_coin_p, tails_coin_p, heads_coin_q, tails_coin_q)
        
    thetas.append((p,r))    
    return thetas


# In[8]:


def e_step(flips, p, r):
    heads_coin_p, tails_coin_p = 0,0
    heads_coin_q, tails_coin_q = 0,0
    for trial in flips:
        llh_p = coin_llh(trial, p)
        llh_q = coin_llh(trial, r)
        p_A = llh_p / (llh_p + llh_q)
        p_B = llh_q / (llh_p + llh_q)
        heads_coin_p += p_A * trial.count("H")
        tails_coin_p += p_A * trial.count("T")
        heads_coin_q += p_B * trial.count("H")
        tails_coin_q += p_B * trial.count("T") 
    return heads_coin_p, tails_coin_p, heads_coin_q, tails_coin_q


# In[9]:


def m_step(heads_coin_p, tails_coin_p, heads_coin_q, tails_coin_q):
    p = heads_coin_p / (heads_coin_p + tails_coin_p)
    r = heads_coin_q / (heads_coin_q + tails_coin_q)
    return p, r


# In[10]:


def coin_llh(flip, bias):
    num_heads = flip.count('H')
    flips_num = len(flip)
    num_tails = flips_num-num_heads
    return pow(bias, num_heads) * pow(1-bias, num_tails)


# In[11]:


thetas = coin_em(flips, 6)


# In[ ]:


thetas

