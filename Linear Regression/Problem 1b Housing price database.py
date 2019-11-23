
# coding: utf-8

# # Regression tree - Housing Price database

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# ## Dara Preprocessing

# - Adding the feature name to the data
# - making sure there are no null values in the data

# In[2]:


word_labels = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "label"]
train_df = pd.read_csv("HousingData/housing_train.txt", delim_whitespace=True, names = word_labels, header=None) 
test_df = pd.read_csv("HousingData/housing_test.txt", delim_whitespace=True, names = word_labels, header=None) 


# In[3]:


# df.head()


# In[4]:


# df['label'].replace(0, 'non-spam',inplace=True)
# df['label'].replace(1, 'spam',inplace=True)
# df.head()


# # Train-Test-Split

# In[5]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indices = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[6]:


random.seed(0)
d_train_df, d_test_df = train_test_split(train_df, test_size=0.20)


# ***

# In[7]:


data = train_df.values
data[:5]


# ### Evaluate data in each bucket

# ### Classify

# In[9]:


def data_classifier(data):
    
    label_column = data[:, -1]
    mean_data = np.mean(label_column)
    classification = mean_data
    
    return classification


# ### Potential splits

# In[10]:


def get_data_partitions(data):
    
    pot_data_partitions = {}
    _, num_columns = data.shape
    for column_index in range(num_columns - 1):        # excluding the last column which is the label
        pot_data_partitions[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                pot_data_partition = (current_value + previous_value) / 2
                
                pot_data_partitions[column_index].append(pot_data_partition)
    
    return pot_data_partitions


# ### Partition the data

# In[11]:


def partition_data(data, split_feature, threshold):
    
    split_feature_values = data[:, split_feature]

    data_left = data[split_feature_values <= threshold]
    data_right = data[split_feature_values >  threshold]
    
    return data_left, data_right


# ### Lowest total variance

# In[12]:


def calculate_variance(data):
    
    label_column = data[:, -1]
    variance = np.var(label_column)
    variance = variance**2/len(data) 
    return variance


# In[13]:


def calculate_total_variance(data_left, data_right):
    
    n = len(data_left) + len(data_right)
    p_data_left = len(data_left) / n
    p_data_right = len(data_right) / n

    total_variance =  (p_data_left * calculate_variance(data_left) 
                      + p_data_right * calculate_variance(data_right))
    
    return total_variance


# In[14]:


def determine_best_split(data, pot_data_partitions):
    
    total_variance = np.inf
    best_split_column = 0
    best_split_value = 0.000
    for column_index in pot_data_partitions:
        for value in pot_data_partitions[column_index]:
            data_left, data_right = partition_data(data, split_feature=column_index, threshold=value)
            current_total_variance = calculate_total_variance(data_left, data_right)

            if current_total_variance <= total_variance:
                total_variance = current_total_variance
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# ## Decision Tree Algorithm

# In[15]:


def decision_tree_algorithm(df, counter=0, min_samples=20, max_depth=10):
    
    global bool_var
    if counter == 0:
        global feature_names
        feature_names = df.columns
        data = df.values
    else:
        data = df
        
    label_column = data[:, -1]
    variance = np.var(label_column)

    if variance == 0:
        bool_var = True
    else:
        bool_var = False
    

    if (bool_var) or (len(data) < min_samples) or (counter == max_depth):
        classification = data_classifier(data)
        
        return classification

    

    else:    
        counter += 1


        pot_data_partitions = get_data_partitions(data)
        split_feature, threshold = determine_best_split(data, pot_data_partitions)
        data_left, data_right = partition_data(data, split_feature, threshold)
        

        feature_name = feature_names[split_feature]
        deciding_factor = "{} <= {}".format(feature_name, threshold)
        sub_tree = {deciding_factor: []}
        

        ans_true = decision_tree_algorithm(data_left, counter, min_samples, max_depth)
        ans_false = decision_tree_algorithm(data_right, counter, min_samples, max_depth)
        

        if ans_true == ans_false:
            sub_tree = ans_true
        else:
            sub_tree[deciding_factor].append(ans_true)
            sub_tree[deciding_factor].append(ans_false)
        
        return sub_tree


# In[16]:


tree = decision_tree_algorithm(train_df)
pprint(tree)


# In[17]:


d_tree = decision_tree_algorithm(d_train_df)
pprint(tree)


# ## Classification
sub_tree = {deciding_factor: [ans_true, ans_false]}
# In[18]:



example = test_df.iloc[0]
example


# In[19]:


def classify_example(example, tree):
    deciding_factor = list(tree.keys())[0]
    feature_name, comparison_operator, value = deciding_factor.split(" ")

    # deciding_factor condition
    if example[feature_name] <= float(value):
        result = tree[deciding_factor][0]
    else:
        result = tree[deciding_factor][1]

    # if result is not dictionary we have reached final step and return the result
    if not isinstance(result, dict):
        return result
    
    # the subtree needs to be divided further
    else:
        return classify_example(example, result)


# In[20]:


classify_example(example, tree)


# ## Calculate mse for test data

# In[21]:


def calculate_mse(df, tree):

    df["prediction"] = df.apply(classify_example, axis=1, args=(tree,))
    
    mse = np.mean((df["label"] - df["prediction"])**2)
    
    return mse


# In[22]:


mse = calculate_mse(test_df, tree)
mse


# In[23]:


mse = calculate_mse(d_test_df, d_tree)
mse


# In[24]:


test_df

