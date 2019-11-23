
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import math


# In[2]:


# df_X = pd.read_csv("../Haar_Features/X_df.csv")
# df_y = pd.read_csv("../Haar_Features/y_df.csv")
# df_X.head()

arr_train = np.load("../haar_train_full.npy")
arr_test = np.load("../haar_test_full.npy")


# In[3]:


train_df = pd.DataFrame(arr_train)
test_df = pd.DataFrame(arr_test)
test_df.head()


# In[4]:


uni_code = {}


# In[5]:


count = 0
while (count!= 10):
    uni_code[count] = np.random.randint(2, size=50)
    count = count+1


# In[6]:


uni_code


# In[7]:


# # df['label'].replace(0, -1,inplace=True)
# # df.head()
# X = np.array(df_X)
# y = np.array(df_y)


# In[8]:


X_train = np.array(train_df.iloc[:12000 , :-1])
y_train = np.array(train_df.iloc[:12000 , -1])

X_test = np.array(test_df.iloc[:6000 , :-1])
y_test = np.array(test_df.iloc[:6000 , -1])


# In[9]:


y_test


# In[10]:


# def train_test_split(df, test_size):
    
#     if isinstance(test_size, float):
#         test_size = round(test_size * len(df))

#     index_list = df.index.tolist()
#     test_indexes = random.sample(population=index_list, k=test_size)

#     test_df = df.loc[test_indexes]
#     train_df = df.drop(test_indexes)
    
#     return train_df, test_df

y_test_mat = [None] * len(y_test)
y_train_mat = [None] * len(y_train)


# In[11]:


# random.seed(3)
# train_df, test_df = train_test_split(df, 0.20)
for i in range(len(y_test)):
    y_test_mat[i] = uni_code.get(int(y_test[i]))
    
for i in range(len(y_train)):
    y_train_mat[i] = uni_code.get(int(y_train[i]))


# In[12]:


# X_train = train_df.iloc[: , :-1]
# y_train = train_df.iloc[: , -1]

# X_test = test_df.iloc[: , :-1]
# y_test = test_df.iloc[: , -1]

y_test_df = pd.DataFrame.from_records(y_test_mat)
X_test_df = pd.DataFrame.from_records(X_test)

y_train_df = pd.DataFrame.from_records(y_train_mat)
X_train_df = pd.DataFrame.from_records(X_train)


# In[13]:


y_train_df = y_train_df.replace(0, -1)
y_test_df = y_test_df.replace(0, -1)
y_train_df.head()


# In[14]:


y_train_mat


# In[15]:


def make_clf_dict():
    clf_dict = {'label': 1, 'feature_name': None, 'threshold': None, 'alpha': None}
    return clf_dict


# In[16]:


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



# In[17]:


iters = 200

def adaboost(X, y):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    classifiers = []
    for i in range(iters):
        clf = make_clf_dict()
        min_err = float('inf')
        a = X_train_df.columns
        column = random.choice(a)
#         for column in X_train.columns:
        feature_values = np.expand_dims(X[column], axis=1)
        unique_values = np.unique(feature_values)
        threshold =  random.choice(unique_values)
#             for threshold in unique_values:
        p = 1
    
        prediction = np.ones(np.shape(y))
             
        prediction[X[column] < threshold] = -1

        error = sum(w[y != prediction])

        if error > 0.5:
            error = 1 - error
            p = -1

#         if error < min_err:
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


# In[18]:


# classifiers = adaboost(X_train, y_train)
# y_pred = predict(X_test)
# y_train_pred = predict(X_train)


y_pred_mat = []
for i in range (50):
    classifiers = adaboost(X_train_df, y_train_df[i])
    y_pred = predict(X_test_df)
    print(y_pred)
    y_pred_mat.append(y_pred)


# In[19]:


# classifiers
y_pred_trans = np.array(y_pred_mat).transpose()


# In[20]:


# np.mean(np.array(y_test) == y_pred)
y_pred_trans[y_pred_trans == -1] = 0


# In[21]:


# testPredict = pd.DataFrame()
# trainPredict = pd.DataFrame()
# testPredict['y_predict'] = y_pred
# trainPredict['y_predict'] = y_train_pred


# testPredict.to_csv('y_test_predict.csv')
# trainPredict.to_csv('y_train_predict.csv')
np.array(y_test_df).shape


# In[22]:


import csv


with open("new_file.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(y_pred_trans)


# In[23]:


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


# In[24]:


y_pred_coded = np.loadtxt(open("new_file.csv", "rb"), delimiter=",")


# In[25]:


y_pred_coded.shape


# In[26]:


y_test = np.squeeze(y_test)


# In[27]:


y_pred_coded = y_pred_coded.astype(int)
x = np.array(y_test).shape[0]


# In[28]:


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
    


# In[29]:


# y_test_list = list()
# for i in range (x):
#     y_test_list.append(y_test[i][0])


# In[30]:


# counter_corr = 0
# for i in range (len(y_pred_decoded)):
#     if (y_pred_decoded[i] == y_test_list[i]):
#         counter_corr += 1


# In[31]:


y_test


# In[32]:


np.array(y_pred_decoded)


# In[33]:


acc = np.mean(y_test == np.array(y_pred_decoded))


# In[34]:


acc

