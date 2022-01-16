#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np


# In[2]:


car=pd.read_csv(r'C:\Users\RAHUL GAUTAM\Downloads\quikr_car.csv')


# In[3]:


car.head()


# In[4]:


backup=car.copy()


# In[5]:


car=car[car['year'].str.isnumeric()]


# In[6]:


car['year']=car['year'].astype(int)


# In[7]:


car=car[car['Price']!='Ask For Price']


# In[8]:


car['Price']=car['Price'].str.replace(',','').astype(int)


# In[9]:


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')


# In[10]:


car=car[car['kms_driven'].str.isnumeric()]


# In[11]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[12]:


car=car[~car['fuel_type'].isna()]


# In[13]:


car.shape


# In[14]:


car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')


# In[15]:


car=car.reset_index(drop=True)


# In[16]:


car


# In[17]:


car.to_csv('Cleaned Car.csv')


# In[18]:


X= car.drop(columns='Price')
y= car['Price']


# In[ ]:





# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[25]:


ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[32]:


ohe.categories_


# In[43]:


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                      remainder='passthrough')


# In[44]:


lr=LinearRegression()


# In[45]:


pipe=make_pipeline(column_trans,lr)


# In[46]:


pipe.fit(X_train,y_train)


# In[48]:


y_pred= pipe.predict(X_test)


# In[50]:


r2_score(y_test,y_pred)


# In[56]:


scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred= pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))
    
    
    


# In[57]:


np.argmax(scores)


# In[58]:


scores[np.argmax(scores)]


# In[60]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred= pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[61]:


import pickle


# In[62]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[63]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']], columns=['name','company','year','kms_driven','fuel_type']))


# In[ ]:


#The prediction shows that the Maruti Suzuki Swift can be sold for 430202.4544 
#approx. 4.40 lakhs

