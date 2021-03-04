#!/usr/bin/env python
# coding: utf-8

# #Naive Bayes Classifier IRIS DATASET

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv("Iris.csv")


# In[3]:


df.head()


# In[4]:


df['Species'].value_counts()


# In[5]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


df.corr()


# In[7]:


sns.heatmap(df.corr())


# In[8]:


sns.countplot('Species',data = df)


# In[9]:


sns.barplot(x='Species',y='SepalLengthCm',data=df)


# In[10]:


sns.boxplot('Species','SepalLengthCm',data=df, palette='rainbow')


# In[11]:


sns.pairplot(df)


# In[ ]:





# In[12]:


# Train_Test Split


# In[13]:


X = df.iloc[:,:4].values
y = df['Species'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[14]:


#Training the Train_set data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[15]:


#Predicting the Test set
y_pred = classifier.predict(X_test) 
y_pred


# In[16]:


#Model Performance/ Accuracy


# In[17]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
conf_matrix


# In[ ]:





# In[ ]:




