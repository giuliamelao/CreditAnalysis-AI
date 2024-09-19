#!/usr/bin/env python
# coding: utf-8

# # Presentation 02
# 
# ## PCA, Pre Processing, Isolation Forest and DBSCAN
# 
# #### The objectives in this notebook are to apply methods of Pre Processing and Linear dimensionality reduction of data and develop methods to easily identify anomalies and outliers values

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


# In[2]:


df = pd.read_csv('/home/gmelao/Desktop/default-of-credit-card-clients.csv')
df.columns = df.iloc[0]
df.drop(0, inplace = True)
df.set_index('ID', inplace = True)
pd.set_option('display.max_columns', 24)
pd.set_option('display.max_rows', 24)


# In[3]:


df = df.apply(lambda df: pd.Series(map(float, df)))


# ###   

# ### Pre Processing
# 
# StandardScaler removes the mean and scales each feature/variable to unit variance. StandardScaler can be influenced by outliers. It's used when the values in the dataset differ greatly or when values are measured in different units of measure.

# In[4]:


scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)


# In[5]:


type(scaled_X)


# ### PCA
# 
# The Principal Component Analysis is a popular unsupervised learning technique for reducing the dimensionality of data. It increases interpretability yet, at the same time, it minimizes information loss. It helps to find the most significant features in a dataset and makes the data easy for plotting in 2D and 3D

# In[6]:


pca_model = PCA(n_components=2)


# In[7]:


pca_results = pca_model.fit_transform(scaled_X)


# In[8]:


type(pca_results)


# In[9]:


plt.scatter(pca_results[:,0], pca_results[:,1])


# ##### MinMax Scaler
# This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

# In[10]:


def scale_data(df):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_arr = scaler.fit_transform(df)
    return scaled_arr


# In[11]:


def vis(arr, labels):
    pca_arr = TSNE(n_components=2, n_jobs=-1).fit_transform(arr)
    df_vis = pd.DataFrame(pca_arr)
    df_vis["labels"] = labels
    sns.scatterplot(data=df_vis, x=0, y=1, hue="labels", palette="Set1")
    plt.savefig("tsne_vis_dbscan.png")
    plt.show()


# In[12]:


def find_outliers(labels):
    display(df.iloc[np.where(labels==-1)])


# In[13]:


scaled_arr = scale_data(df)


# ### DBSCAN
# 
# Density-Based Spatial Clustering of Applications with Noise, it's a clustering method. Finds core samples of high density and expands clusters from them.

# In[14]:


def training_data_DBSCAN(arr):
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(PCA(n_components=0.9).fit_transform(scaled_arr))
    return labels


# In[15]:


lab = training_data_DBSCAN(scaled_arr)
vis(scaled_arr, lab)
find_outliers(lab)


# ### Isolation Forest
# 
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# 
#     

# In[16]:


def training_data_IF(arr, scaled_df):
    clf = IsolationForest(random_state=0).fit(arr)
    pca = PCA(n_components=0.9)
    pca_arr = pca.fit_transform(arr)
    clf.fit(pca_arr)
    labels = clf.predict(pca_arr)
    return clf, labels, pca


# In[17]:


clf, lab, pca = training_data_IF(scaled_arr, scaled_arr)


# In[18]:


scaler = MinMaxScaler(feature_range=(-1,1))
scaled_arr = scaler.fit_transform(df)
dbscan = DBSCAN()
labels = dbscan.fit_predict(PCA(n_components=0.9).fit_transform(scaled_arr))


# In[50]:


l = []
for _ in range(df.shape[1]):
    l.append(random.random() * 2 - 1)
l


# In[51]:


if clf.predict(pca.transform([l])) == -1:
    #print(l)
    print("ANOMALOUS")


# ### TSNE
# 
# TSNE is a tool to visualize high-dimensional data. It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high.

# In[19]:


tsne_arr = TSNE(n_components=2, n_jobs=-1).fit_transform(scaled_arr)
tsne_arr


# In[20]:


df_vis = pd.DataFrame(tsne_arr)
df_vis


# In[21]:


df_vis["labels"] = labels


# In[22]:


df_vis


# In[27]:


sns.scatterplot(data=df_vis, x=0, y=1, hue="labels", palette="Set1")


# ![tsne.png](./tsne.png)

# ##   

# In[ ]:





# In[ ]:





# In[45]:


#df_pca = pd.DataFrame(PCA(n_components=0.8).fit_transform(df))
#display(df_pca)

#sns.scatterplot(df_pca, x=0, y=1)#, hue="AGE")


# In[28]:


type(df_vis['labels'])


# In[29]:


labels_out = df_vis.loc[df_vis['labels'] == -1]
labels_out


#        

# The function below generates a synthetic row to validate if it's anomalous or not.

# In[ ]:





# In[ ]:





# In[41]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




