#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display


# In[13]:


X, _ = make_blobs(
    n_samples=750, n_features=10, cluster_std=0.4, random_state=0
)
X = StandardScaler().fit_transform(X)


# In[14]:


df = pd.DataFrame(X)
display(df)


# In[15]:


dbscan = DBSCAN()
clustering = dbscan.fit_predict(X)


# In[24]:


np.unique(clustering)


# In[16]:


sns.pairplot(df)


# In[23]:


df_pca = pd.DataFrame(PCA(n_components=0.9).fit_transform(df))
display(df_pca)
df_with_labels = df_pca.copy()
df_with_labels["clusters"] = clustering
display(df_with_labels)
sns.scatterplot(df_with_labels, x=0, y=1, hue="clusters")

# # %%
# clustering.labels_

