
# coding: utf-8

# In[1]:


from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np


# In[2]:


X = np.array(pd.read_csv("../Haar_Features/X_df.csv") )
y = np.array( pd.read_csv("../Haar_Features/y_df.csv") )


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 


from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
accuracy = svm_model_linear.score(X_test, y_test) 
  
cm = confusion_matrix(y_test, svm_predictions) 


# In[5]:


accuracy 


# In[ ]:


np.mean(svm_predictions == y_test)

