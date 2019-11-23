
# coding: utf-8

# # Decision Tree for classification data - Spambase

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# ## Data Preprocessing

# - Adding the feature name to the data
# - making sure there are no null values in the data

# In[2]:


word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
df = pd.read_csv("spambase/spambase.data", names = word_labels, header=None) 


# In[3]:


df.head()


# In[4]:


df['label'].replace(0, 'non-spam',inplace=True)
df['label'].replace(1, 'spam',inplace=True)
df.head()


# ## Train-Test-Split

# In[5]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indexes = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    
    return train_df, test_df


# In[6]:


from random import randrange
def make_kfolds(df, kfolds):
    data_folds = list()
    df_list = df.values.tolist()
    fold_size = int(len(df) / kfolds)
    for i in range(kfolds):
        fold = list()
        while len(fold) < fold_size:
            x = len(df_list);
            index = randrange(x)
            fold.append(df_list.pop(index))
        data_folds.append(pd.DataFrame(fold))
    return data_folds


# In[7]:


# random.seed(0)
# train_df, test_df = train_test_split(df, test_size=0.2)


# In[8]:


data_folds = make_kfolds(df, 5)
data_folds[4]


# ***

# In[9]:


# # converting the data frame to numpy array to increase the computation speed
# data = train_df.values
# data[:5]


# ### Evaluate data in each buckect

# ### Data classifier

# In[10]:


#classify the data in the bucket after split into the label based on the maximum number of dataobjects in that bucket.
def data_classifier(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# ### Potential splits

# In[11]:


#function returns the potential splits, which contains the datapoints between the splits.
def get_data_partitions(data):
    
    pot_data_partitions = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):        # excluding the last column which is the label
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


# ### Partition the Data 

# In[12]:


#On finding the best split feature and best split point in that feature, it divides the data into two buckets 
def partition_data(data, split_feature, threshold):
    
    split_feature_values = data[:, split_feature]

    data_left = data[split_feature_values <= threshold]
    data_right = data[split_feature_values >  threshold]
    
    return data_left, data_right


# ### Lowest Overall Entropy

# In[13]:


def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


# In[14]:


def calc_total_entropy(data_left, data_right):
    
    n = len(data_left) + len(data_right)
    p_data_left = len(data_left) / n
    p_data_right = len(data_right) / n

    total_entropy =  (p_data_left * calculate_entropy(data_left) 
                      + p_data_right * calculate_entropy(data_right))
    
    return total_entropy


# In[15]:


def determine_best_partition(data, pot_data_partitions):
    
    total_entropy = np.inf
    for column_index in pot_data_partitions:
        for value in pot_data_partitions[column_index]:
            data_left, data_right = partition_data(data, split_feature=column_index, threshold=value)
            current_total_entropy = calc_total_entropy(data_left, data_right)

            if current_total_entropy <= total_entropy:
                total_entropy = current_total_entropy
                best_split_feature = column_index
                best_split_threshold = value
    
    return best_split_feature, best_split_threshold


# ## Decision Tree Algorithm

# In[16]:


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    
    global bool_var
    if counter == 0:
        global feature_names
        feature_names = df.columns
        data = df.values
    else:
        data = df           
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        bool_var = True
    else:
        bool_var = False
    

    if (bool_var) or (len(data) < min_samples) or (counter == max_depth):
        classification = data_classifier(data)
        
        return classification

    

    else:    
        counter += 1

        pot_data_partitions = get_data_partitions(data)
        split_feature, threshold = determine_best_partition(data, pot_data_partitions)
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


# In[17]:


def create_labels(df):
    word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", "char_freq1", "char_freq2", "char_freq3", 
              "char_freq4", "char_freq5", "char_freq6", "cap_run_length_avg", "cap_run_length_longest", "cap_run_length_total", "label"]
    df.columns = word_labels
    return df


# In[18]:


print(len(data_folds))


# In[21]:


# tree = decision_tree_algorithm(train_df)
# pprint(tree)

def make_trees(data_folds):
    trees = list()
    for i in range(len(data_folds)):
        df_new = pd.DataFrame()
        df_new_test = pd.DataFrame()
        for j in range(len(data_folds)):
            if(j != i):
                df_new = df_new.append(data_folds[j])
        df_new_train = create_labels(df_new)
        tree = decision_tree_algorithm(df_new_train)
        trees.append(tree)
    return trees


# In[22]:


trees = make_trees(data_folds)


# In[23]:


def make_tests(data_folds):
    tests = list()
    for k in range(len(data_folds)):
        df_new_test = create_labels(data_folds[k])
        tests.append(df_new_test)
    return tests


# In[24]:


tests = make_tests(data_folds)


# In[35]:


trees[3]


# In[26]:


tests[3]


# ## Classification
sub_tree = {deciding_factor: [ans_true, ans_false]}
# In[27]:


example = tests[0].iloc[0]
example


# In[28]:


def classify_example(example, tree):
    deciding_factor = list(tree.keys())[0]
    feature_name, comparison_operator, value = deciding_factor.split(" ")


    if example[feature_name] <= float(value):
        result = tree[deciding_factor][0]
    else:
        result = tree[deciding_factor][1]


    if not isinstance(result, dict):
        return result
    

    else:
        residual_tree = result
        return classify_example(example, residual_tree)


# In[29]:


classify_example(example, trees[0])


# ## Calculate Accuracy

# In[30]:


def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["boolean_label_correctness"] = df["classification"] == df["label"]
    
    accuracy = df["boolean_label_correctness"].mean()
    
    return accuracy


# In[31]:


accuracy = calculate_accuracy(tests[2], trees[2])
accuracy


# In[32]:


def calculate_average_accuracy(tests, trees):
    k_accuracy = list()
    for i in range(len(tests)):
        accuracy = calculate_accuracy(tests[i], trees[i])
        k_accuracy.append(accuracy) 
    return sum(k_accuracy) / len(k_accuracy) 
        


# In[33]:


calculate_average_accuracy(tests, trees)

