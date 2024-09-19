#!/usr/bin/env python
# coding: utf-8

# # DS Clustering / K-Means

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


df = pd.read_csv('/home/gmelao/Desktop/default-of-credit-card-clients.csv')
df.columns = df.iloc[0]
df.drop(0, inplace = True)
df.set_index('ID', inplace = True)
pd.set_option('display.max_columns', 24)
pd.set_option('display.max_rows', 24)


# In[18]:


df = df.apply(lambda df: pd.Series(map(float, df)))


# In[19]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df
# In[33]:


df.columns


# In[20]:


scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)


# In[21]:


model = KMeans(n_clusters=5)
cluster_label = model.fit_predict(scaled_df)


# In[22]:


cluster_label


# In[23]:


df['Cluster'] = cluster_label


# In[24]:


df.corr()['Cluster'].iloc[:-1].sort_values().plot(kind='bar')


# In[25]:


df.corr()['LIMIT_BAL'].iloc[:-1].sort_values().plot(kind='bar')


# In[11]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 100)
pred_y = kmeans.fit_predict(df)
plt.scatter(df['LIMIT_BAL'], df['Cluster'], c = pred_y)
plt.scatter(kmeans.cluster_centers_,kmeans.cluster_centers_, s = 70, c = 'red')
plt.show()


# In[30]:


def display_models(model, data, X, Y):
    labels = model.fit_predict(data)
    sns.scatterplot(data=data, x=X, y=Y, hue=labels)


# In[31]:


model = KMeans(n_clusters=4)


# In[32]:


display_models(model, df, 'LIMIT_BAL', 'AGE')


# In[34]:


display_models(model, df, 'LIMIT_BAL', 'Cluster')


# In[35]:


df


# In[ ]:




