#!/usr/bin/env python
# coding: utf-8

# Adding all the libraries necessary for the project

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


# In[4]:


os.listdir(os.getcwd())
df=pd.read_csv('life.csv')


# In[39]:


df.head(3)


# In[40]:


df.isnull().sum()


# In[41]:


print(len(df))


# In[42]:


df.describe()


# In[43]:


df.dtypes


# In[44]:


df=df.dropna(axis=0)


# In[45]:


len(df)


# In[46]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[54]:


r=df.groupby('Country').count()
print(r)


# In[55]:


r=df.groupby('Status').count()
print(r)


# In[58]:


df.plot(x='Status', y=['Adult Mortality'], figsize=(10,5), grid=True)


# In[ ]:





# In[61]:


df.columns=df.columns.str.replace(' ','_')


# In[63]:


df.columns


# In[75]:


plt.scatter(df.Schooling, df.Life_expectancy_)
plt.show()


# In[92]:


import seaborn as sns
plt.figure(figsize=(15,10))
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[94]:


plt.scatter(df.Income_composition_of_resources, df.Life_expectancy_)
plt.show()


# In[95]:


plt.scatter(df._BMI_, df.Life_expectancy_)
plt.show()


# In[96]:


plt.scatter(df.Alcohol, df.Life_expectancy_)
plt.show()


# In[97]:


plt.scatter(df.Adult_Mortality, df.Life_expectancy_)
plt.show()


# In[120]:


X = pd.DataFrame(np.c_[df['Schooling'], df['Adult_Mortality']], columns = ['Income_composition_of_resources','Schooling'])
Y = df['Life_expectancy_']


# In[121]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)


# In[122]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[123]:


y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)


# In[124]:


print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




