#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[29]:


import numpy as np


# In[36]:


import matplotlib.pyplot as plt


# In[37]:


import seaborn as sns


# In[38]:


df= pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv')


# In[39]:


df.head()


# In[40]:


df.info()


# In[41]:


df.describe()


# In[42]:


df.columns


# In[43]:


df.shape


# In[45]:


df.nunique()


# In[46]:


df=df.dropna()


# In[47]:


df.info()


# In[50]:


X=df[['displacement', 'horsepower', 'weight',
       'acceleration']]
y=df['mpg']


# In[52]:


X.shape


# In[53]:


y.shape


# In[54]:


X


# In[55]:


y


# In[56]:


from sklearn.preprocessing import StandardScaler


# In[57]:


ss= StandardScaler()


# In[58]:


X=ss.fit_transform(X)


# In[59]:


X


# In[60]:


pd.DataFrame(X).describe()


# In[61]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[62]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


lr= LinearRegression()


# In[65]:


lr.fit(X_train,y_train)


# In[66]:


lr.intercept_


# In[67]:


lr.coef_


# In[68]:


y_pred=lr.predict(X_test)


# In[69]:


y_pred


# In[70]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# In[71]:


mean_absolute_error(y_test,y_pred)


# In[72]:


mean_absolute_percentage_error(y_test,y_pred)


# In[73]:


mean_squared_error(y_test,y_pred)


# In[ ]:




