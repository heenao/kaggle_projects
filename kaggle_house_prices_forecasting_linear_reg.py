#!/usr/bin/env python
# coding: utf-8

# # Kaggle house price forecasting competition
# 
# 
# #### https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# #### https://www.dataquest.io/blog/kaggle-getting-started/

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[ ]:


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[2]:


train = pd.read_csv("house_prices_train.csv")
test = pd.read_csv("house_prices_test.csv")


# In[3]:


train


# In[7]:


train.columns


# In[8]:


train.SalePrice.describe()


# In[9]:


print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()


# In[10]:


target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


# In[11]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features


# In[12]:


numeric_features.dtypes


# In[14]:


corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])


# In[15]:


metrics = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','LotArea','LotFrontage','YearBuilt']


# In[20]:


sns.pairplot(train[metrics ])#, hue= 'mediatype')


# In[21]:


train.OverallQual.unique()


# In[22]:


train.GarageCars.unique()


# In[23]:


quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice',
                                  aggfunc=np.median)
quality_pivot


# In[25]:


quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[26]:


plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()


# In[28]:


train = train[train['GarageArea'] < 1200]


# In[29]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls


# In[30]:


train.MiscFeature.unique()


# In[31]:


categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()


# In[32]:


train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)


# In[34]:


print ('Encoded: \n')
print (train.enc_street.value_counts())


# In[37]:


condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[38]:


def encode(x):
 return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


# In[39]:


condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[40]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()
data


# In[41]:


## Build Model


# In[42]:


y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)


# In[52]:


lr = linear_model.LinearRegression()


# In[54]:


model = lr.fit(X_train, y_train)


# In[55]:


print ("R^2 is: \n", model.score(X_test, y_test))


# In[56]:


predictions = model.predict(X_test)
predictions


# In[57]:


print ('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[59]:


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




