#!/usr/bin/env python
# coding: utf-8

# # Data Science Training​
# ## Default of credit card clients Data Set Project​

# #### Upload data set

# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale 


# In[2]:


df = pd.read_csv('/home/gmelao/Desktop/default-of-credit-card-clients.csv')
df.columns = df.iloc[0]
df.drop(0, inplace = True)
df.set_index('ID', inplace = True)
pd.set_option('display.max_columns', 24)
pd.set_option('display.max_rows', 24)


# #### Features and types

# In[3]:


df.dtypes


# ###  

# ### Useful Info
# 
# Gender (1 = male; 2 = female)
# 
# Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# 
# Marital status (1 = married; 2 = single; 3 = others).
# 
# The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# 

# ##  

# In[4]:


df.head()


# In[5]:


df[['BILL_AMT1', 'PAY_AMT1']].head()


# In[6]:


df[['BILL_AMT2', 'PAY_AMT2']].head()


#   

# #### FIrst data treatment

# In[7]:


df = df.apply(lambda df: pd.Series(map(float, df)))


# In[8]:


df.dtypes


#   

# In[9]:


df_no_cats = df.drop(['SEX', 'MARRIAGE', 'EDUCATION', 'default payment next month'], axis=1)
df_no_cats.describe()


# In[10]:


df_cats = df[['SEX', 'MARRIAGE', 'EDUCATION', 'default payment next month']].astype('category')
df_cats.describe()


#     

#    

# ### Data Quality

# #### Uniqueness
# Verify if duplicated values exists
# 

# In[13]:


df2 = df_no_cats.apply(lambda df: df.duplicated(), axis=1)
df2.sum()


# In[62]:


df.groupby('PAY_6').size()


# #### Completeness
# Show how many null values in the data set

# In[14]:


df.isna().sum()


# 

#      

# ## Boxplot
# 
# It's a method for graphically demonstrating the variation groups of numerical data through their quartiles. Boxplot graphs are useful to identify dospersion of data, simetry, outliers and positions.

# In[69]:


for c in df_no_cats.columns:
    sns.boxplot(df_no_cats, x=c)
    plt.show()
    plt.close()


# In[70]:


for c in df_no_cats.columns:
    ax = df_no_cats.boxplot(c)
    if c.startswith('BILL_AMT') or c.startswith('PAY_AMT'):
        ax.set_yscale("log")
        plt.show()
        plt.close()


#      

#      

# ## Histograms
# 
# This graph will show the frequency distributions, it will be possible to identify if each feature is a Gaussian distribution or not
# 

# In[67]:


for c in df_no_cats.columns:
    ax = sns.histplot(df_no_cats, x=c)
    if c.startswith('BILL_AMT') or c.startswith('PAY_AMT'):
        ax.set_yscale("log")
    plt.show()
    plt.close()


#           
#           

# ## Plotting a diagonal correlation matrix
# 
# This diagram will show the correlation of each feature individually
# 

# In[32]:


corr = df_no_cats.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[ ]:





# ### Dummies Variables and Categorical Data
# 
# Dummy variables enable us to use a single regression equation to represent multiple groups
# 

# In[38]:


pd.get_dummies(df_cats.drop(['default payment next month'], axis=1)).head()


# In[42]:


df2 = pd.concat([df_no_cats, pd.get_dummies(df_cats.drop(['default payment next month'], axis=1))], axis=1)
df2.head()


# In[43]:


df2.describe()


# In[ ]:




