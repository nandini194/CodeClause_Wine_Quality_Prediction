#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wine = pd.read_csv('WineQT.csv')
wine.head()


# In[3]:


wine.describe()


# In[4]:


wine.info()


# In[5]:


fig = plt.figure(figsize=(15,10))

plt.subplot(3,4,1)
sns.barplot(x='Quality',y='Fixed_Acidity',data=wine)

plt.subplot(3,4,2)
sns.barplot(x='Quality',y='Volatile_Acidity',data=wine)

plt.subplot(3,4,3)
sns.barplot(x='Quality',y='Citric_Acid',data=wine)

plt.subplot(3,4,4)
sns.barplot(x='Quality',y='Residual_Sugar',data=wine)

plt.subplot(3,4,5)
sns.barplot(x='Quality',y='Chlorides',data=wine)

plt.subplot(3,4,6)
sns.barplot(x='Quality',y='Sulfur_Dioxide',data=wine)

plt.subplot(3,4,7)
sns.barplot(x='Quality',y='Total_Sulfur_Dioxide',data=wine)

plt.subplot(3,4,8)
sns.barplot(x='Quality',y='Density',data=wine)

plt.subplot(3,4,9)
sns.barplot(x='Quality',y='pH',data=wine)

plt.subplot(3,4,10)
sns.barplot(x='Quality',y='Sulphates',data=wine)

plt.subplot(3,4,11)
sns.barplot(x='Quality',y='Alcohol',data=wine)

plt.tight_layout()


# In[6]:


wine['Quality'].value_counts()


# In[7]:


ranges = (2,6.5,8) 
groups = ['bad','good']
wine['Quality'] = pd.cut(wine['Quality'],bins=ranges,labels=groups)


# In[8]:


le = LabelEncoder()
wine['Quality'] = le.fit_transform(wine['Quality'])
wine.head()


# In[9]:


wine['Quality'].value_counts()


# In[10]:


good_quality = wine[wine['Quality']==1]
bad_quality = wine[wine['Quality']==0]

bad_quality = bad_quality.sample(frac=1)
bad_quality = bad_quality[:217]

new_df = pd.concat([good_quality,bad_quality])
new_df = new_df.sample(frac=1)
new_df


# In[11]:


new_df['Quality'].value_counts()


# In[12]:


new_df.corr()['Quality'].sort_values(ascending=False)


# In[13]:


from sklearn.model_selection import train_test_split

X = new_df.drop('Quality',axis=1) 
y = new_df['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[14]:


param = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000]}

grid_rf = GridSearchCV(RandomForestClassifier(),param,scoring='accuracy',cv=10,)
grid_rf.fit(X_train, y_train)

print('Best parameters --> ', grid_rf.best_params_)

pred = grid_rf.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print(accuracy_score(y_test,pred))


# In[ ]:




