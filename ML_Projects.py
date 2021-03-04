#!/usr/bin/env python
# coding: utf-8

# #Naive Bayes Classifier IRIS DATASET

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


df= pd.read_csv("Iris.csv")


# In[45]:


df.head()


# In[46]:


sns.countplot('Species',data = df)


# In[47]:


sns.barplot(x='Species',y='SepalLengthCm',data=df)


# In[48]:


df['Species'].value_counts()


# In[27]:


# Train_Test Split


# In[49]:


X = df.iloc[:,:4].values
y = df['Species'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# In[50]:


#Training the Train_set data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[51]:


#Perdicting the Test Test
y_pred = classifier.predict(X_test) 
y_pred


# In[38]:


#Model Performance/ Accuracy


# In[52]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
conf_matrix


# In[ ]:




