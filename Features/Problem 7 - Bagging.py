
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
from __future__ import division, print_function
import math
from random import seed


# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("../spambase/spambase.data", names = word_labels, header=None) 
df.head()


# In[3]:


# def train_test_split(df, test_size):
    
#     if isinstance(test_size, float):
#         test_size = round(test_size * len(df))

#     index_list = df.index.tolist()
#     test_indexes = random.sample(population=index_list, k=test_size)

#     test_df = df.loc[test_indexes]
#     train_df = df.drop(test_indexes)
    
#     return train_df, test_df


# In[4]:


# random.seed(3)
# train_df, test_df = train_test_split(df, 0.20)


# In[5]:


# X_train = train_df.iloc[: , :-1]
# y_train = train_df.iloc[: , -1]

# X_test = test_df.iloc[: , :-1]
# y_test = test_df.iloc[: , -1]


# In[9]:


def make_k_folds(df, kfolds):
    df_split = list()
    df_copy =  df.vals.tolist()
    fold_size = int(len(df) / kfolds)
    for i in range(kfolds):
        fold = list()
        while len(fold) < fold_size:
            x = len(df_copy)
            sample_ind = random.randrange(x)
            fold.append(df_copy.pop(sample_ind))
        df_split.append(fold)
    return df_split
 


# In[10]:


def calc_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 


# In[11]:


def eval_algo(n_trees):
    folds = make_k_folds(df, kfolds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = bagging(train_set, test_set, max_depth, min_size, sample_size, n_trees)
        actual = [row[-1] for row in fold]
        accuracy = calc_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores
 


# In[12]:


def test_split(sample_ind, val, df):
    left, right = list(), list()
    for row in df:
        if row[sample_ind] < val:
            left.append(row)
        else:
            right.append(row)
    return left, right
 


# In[13]:


def calc_entropy(grps, classes):
    n_instances = float(sum([len(grp) for grp in grps]))
    entropy_val = 0.0
    for grp in grps:
        size = float(len(grp))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in grp].count(class_val) / size
            score += p * p
        entropy_val += (1.0 - score) * (size / n_instances)
    return entropy_val


# In[14]:


def get_split(df):
   class_vals = list(set(row[-1] for row in df))
   p_sample_ind, p_val, p_score, p_grps = np.inf, np.inf, np.inf, None
   for sample_ind in range(len(df[0])-1):
       for row in df:
           grps = test_split(sample_ind, row[sample_ind], df)
           entropy_val = calc_entropy(grps, class_vals)
           if entropy_val < p_score:
               p_sample_ind, p_val, p_score, p_grps = sample_ind, row[sample_ind], entropy_val, grps
   return {'sample_ind':p_sample_ind, 'val':p_val, 'grps':p_grps}


# In[15]:


def count_res(grp):
    results = [row[-1] for row in grp]
    return max(set(results), key=results.count)
 


# In[16]:


def split(node, max_depth, min_size, depth):
    left, right = node['grps']
    del(node['grps'])
    if not left or not right:
        node['left'] = node['right'] = count_res(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = count_res(left), count_res(right)
        return
    if len(left) <= min_size:
        node['left'] = count_res(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = count_res(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)
 


# In[17]:


def make_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# In[18]:


def predict(node, row):
    if row[node['sample_ind']] < node['val']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
 


# In[19]:


def subsample(df, ratio):
    sample = list()
    n_sample = round(len(df) * ratio)
    while len(sample) < n_sample:
        sample_ind = random.randrange(len(df))
        sample.append(df[sample_ind])
    return sample


# In[20]:


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# In[21]:


def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = make_tree(sample, max_depth, min_size)
        trees.append(tree)
        predictions = [predict(tree, row) for tree in trees]
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)
 


# In[ ]:


kfolds = 5
max_depth = 6
min_size = 2
sample_size = 0.50
for n_trees in [1, 5, 10]:
    scores = eval_algo(n_trees)
    print('Trees: ' % n_trees)
    print('Scores: ' % scores)
    print('Mean Accuracy: ' % (sum(scores)/float(len(scores))))

