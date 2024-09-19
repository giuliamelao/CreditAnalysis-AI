#!/usr/bin/env python
# coding: utf-8

# # DS Random Forest/Decision Trees

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[8]:


df = pd.read_csv('/home/gmelao/Desktop/default-of-credit-card-clients.csv')
df.columns = df.iloc[0]
df.drop(0, inplace = True)
df.set_index('ID', inplace = True)
pd.set_option('display.max_columns', 24)
pd.set_option('display.max_rows', 24)


# In[9]:


df2 = df.apply(lambda df: pd.Series(map(float, df)))


# In[15]:


df2.columns


# ### Split Train Test - Decision Tree

# In[4]:


X = df2
y = df2['LIMIT_BAL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[5]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[6]:


pd.DataFrame(index=X.columns, data = model.feature_importances_, columns=['Feature Importance']).sort_values('Feature Importance')


# In[7]:


def report_model(model): 
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print('\n')
    plt.figure(figsize=(40,40), dpi=600)
    plot_tree(model, feature_names = X.columns, filled=True);


# In[8]:


report_model(model)


# In[9]:


pruned_tree = DecisionTreeClassifier(max_depth = 2)
pruned_tree.fit(X_train, y_train)
report_model(pruned_tree)


# ### Random Forest

# In[10]:


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[11]:


preds2 = rfc.predict(X_test)


# In[12]:


pd.DataFrame(index=X.columns, data = rfc.feature_importances_, columns=['RFC Feature Importance']).sort_values('RFC Feature Importance')


# In[16]:


sns.pairplot(df2, hue='LIMIT_BAL', x_vars='LIMIT_BAL')


# In[13]:


sns.pairplot(df2, hue='LIMIT_BAL')


# ### GridSearch

# In[14]:


X = df2.drop('LIMIT_BAL', axis=1)
y = df2['LIMIT_BAL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

n_estimators = [200]
max_features = [4]
bootstrap = [True]
oob_score = [False]
param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             'oob_score':oob_score}


# In[15]:


grid = GridSearchCV(rfc, param_grid)
grid.fit(X_train, y_train)


# In[16]:


#grid.best_params_


# In[17]:


report_model(grid)


# In[ ]:


errors = []
misclassifications = []

for n in range(1,200):
    rfc = RandomForestClassifier(n_estimators=n, max_features=4)
    rfc.fit(X_train, y_train)
    preds = rfc.predict(X_test)
    err = 1 - accuracy_score(y_test, preds)
    missed = np.sum(preds != y_test)
    
    errors.append(err)
    missclassifications.append(missed)


# In[ ]:


plt.plot(range(1,200), errors)


# In[ ]:




