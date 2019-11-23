
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import math


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


df['label'].replace(0, -1,inplace=True)
df.head()


# In[4]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    index_list = df.index.tolist()
    test_indexes = random.sample(population=index_list, k=test_size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    
    return train_df, test_df


# In[5]:


random.seed(3)
train_df, test_df = train_test_split(df, 0.20)


# In[6]:


X_train = train_df.iloc[: , :-1]
y_train = train_df.iloc[: , -1]

X_test = test_df.iloc[: , :-1]
y_test = test_df.iloc[: , -1]


# In[7]:


def make_clf_dict():
    clf_dict = {'label': 1, 'feature_name': None, 'threshold': None, 'alpha': None}
    return clf_dict


# In[8]:


pred_array = []
def predict(X):
    n_samples = np.shape(X)[0]
    y_pred = np.zeros((n_samples, 1))
    for clf in classifiers:
        predictions = np.ones(np.shape(y_pred))
        val = clf['label'] * X[clf['feature_name']]
        c = X[clf['feature_name']]
        index_neg = (clf['label'] * X[clf['feature_name']] < clf['label'] * clf['threshold'])
        predictions[index_neg] = -1
        y_pred += clf['alpha'] * predictions
        y_pred_flatten = y_pred.flatten()
        y_pred_reshape = y_pred_flatten.reshape(-1, 1)
        pred_array.append(y_pred_reshape)
    y_pred = np.sign(y_pred).flatten()

    return y_pred



# In[9]:


iters = 1000

def adaboost(X, y):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    classifiers = []
    for i in range(iters):
        clf = make_clf_dict()
        min_err = float('inf')
        for column in X_train.columns:
            feature_values = np.expand_dims(X[column], axis=1)
            unique_values = np.unique(feature_values)
 
            for threshold in unique_values:
                p = 1
    
                prediction = np.ones(np.shape(y))
             
                prediction[X[column] < threshold] = -1

                error = sum(w[y != prediction])

                if error > 0.5:
                    error = 1 - error
                    p = -1

                if error < min_err:
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


# In[10]:


# def calc_Hx(X, classifiers):
#     n_samples, n_features = np.shape(X)
#     pred_array = np.zeros(len(classifiers), n_samples)
#     for i in range n_samples:
#         for col in X.columns:
#             counter b = 0
#             for clf in classifiers:
#                 pred_array[i][b] = clf['label'] * X[clf['feature_name']]
            


# In[11]:


classifiers = adaboost(X_train, y_train)
y_pred = predict(X_test)



# In[12]:


classifiers


# In[13]:


len(y_test)


# In[14]:


np.mean(np.array(y_test) == y_pred)


# In[15]:


# testPredict = pd.DataFrame()
# trainPredict = pd.DataFrame()
# testPredict['y_predict'] = y_pred
# trainPredict['y_predict'] = y_train_pred


# testPredict.to_csv('y_test_predict.csv')
# trainPredict.to_csv('y_train_predict.csv')


# In[16]:


dict = {'Name': 5, 'Age': 7}
dict2 = {'Name' : 6, 'Age': 3}
x = dict['Name'] * dict2['Name']
dict['Sex'] = x


# In[17]:


y_test = y_test.tolist()
num_cols = len(pred_array[0])
margins = np.zeros((X_test.shape[1], num_cols))
numerator = {}
denominator = {}
for i in range(len(pred_array[0])):
    for k in range(len(classifiers)):
        column_name = classifiers[k]['feature_name']
        if column_name in numerator:
            numerator[column_name] += classifiers[k]['alpha'] * pred_array[k][i]
        else:
            numerator[column_name] = classifiers[k]['alpha'] * pred_array[k][i]
        if column_name in denominator:
            denominator[column_name] += abs(classifiers[k]['alpha'])
        else:
            denominator[column_name] = abs(classifiers[k]['alpha'])
        
    for j in range(X_test.shape[1]):
        col_name = word_labels[j]
        if col_name in numerator and col_name in denominator:
            margins[j][i] = (y_test[i] * numerator[col_name]) / denominator[col_name]
        


# In[18]:


margin_avg = {}
for i in range(margins.shape[0]):
    margin_avg[i] = sum(margins[i])/ len(margins[i])


# In[19]:


# for i in range(len(y_test)):
    


# In[20]:


# len(pred_array[0])


# In[21]:


X_test.shape[1]


# In[22]:


margin_avg


# In[23]:


import operator
sorted_d = sorted(margin_avg.items(), key=operator.itemgetter(1))


# In[24]:


sorted_d


# In[ ]:


52, 51, 56, 15, 6, 22, 23, 4, 26, 24, 7, 54, 5, 19, 18.

