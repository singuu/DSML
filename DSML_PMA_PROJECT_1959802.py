#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np

#import dataset
df = pd.read_csv("PMA_blockbuster_movies.csv")
df_original = pd.read_csv("PMA_blockbuster_movies.csv")


#-----------Data Cleaning & Feature Engineering-----------------

# Data cleaning
# drop missing value
df = df.dropna(axis=0,subset = ["poster_url"])

# drop useless column
df = df.drop(columns=['poster_url','2015_inflation','genres','title','worldwide_gross'])

# Feature engineering
# Data transformation
# genre - each genre has a feature
df = pd.get_dummies(df, columns=["Genre_1"], prefix=["genre"])
df = pd.get_dummies(df, columns=["Genre_2"], prefix=["genre"])
df = pd.get_dummies(df, columns=["Genre_3"], prefix=["genre"])

# rating - each type has a feature
df = pd.get_dummies(df, columns=["rating"], prefix=["rating"])

# release_date - change into release_month and each month has a feature
df['release_date'] = pd.to_datetime(df['release_date']) #Change datetime typ
df['release_date'] = pd.DatetimeIndex(df['release_date']).month
df = pd.get_dummies(df, columns=["release_date"], prefix=["release_month"])

# year - change into three intervals
bins=[df['year'].min()-1,2000,2010,df['year'].max()+1]
labels=['-1999','2000-2009','2010-2014']

df['year_category']=pd.cut(
        df['year'],
        bins,
        right=False,
        labels=labels)
df = pd.get_dummies(df, columns=["year_category"], prefix=["year_category"])
df = df.drop(columns=['year'])

# studio - add some features about some famous studios
# insert column 'studio_isColumbia'
def isColumbia(studio):
    if 'columbia' in studio or 'Columbia' in studio:
        return 1
    else:
        return 0
df['studio_isColumbia'] = df['studio'].map(lambda x: isColumbia(x))

# insert column 'studio_isDisney'
def isDisney(studio):
    if 'Disney' in studio:
        return 1
    else:
        return 0
df['studio_isDisney'] = df['studio'].map(lambda x: isDisney(x))

# insert column 'studio_isFox'
def isFox(studio):
    if 'fox' in studio or 'Fox' in studio or 'FOX' in studio:
        return 1
    else:
        return 0
df['studio_isFox'] = df['studio'].map(lambda x: isFox(x))

# insert column 'studio_isMarvel'
def isMarvel(studio):
    if 'Marvel' in studio:
        return 1
    else:
        return 0
df['studio_isMarvel'] = df['studio'].map(lambda x: isMarvel(x))

# insert column 'studio_isParamount'
def isParamount(studio):
    if 'Paramount' in studio:
        return 1
    else:
        return 0
df['studio_isParamount'] = df['studio'].map(lambda x: isParamount(x))

# insert column 'studio_isUniversal'
def isUniversal(studio):
    if 'Universal' in studio:
        return 1
    else:
        return 0
df['studio_isUniversal'] = df['studio'].map(lambda x: isUniversal(x))

# insert column 'studio_isWarner'
def isWarner(studio):
    if 'Warner' in studio:
        return 1
    else:
        return 0
df['studio_isWarner'] = df['studio'].map(lambda x: isWarner(x))

df = df.drop(columns=['studio'])

# Data standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_vars = ['rt_audience_score','rt_freshness','imdb_rating','length']
df[num_vars] = scaler.fit_transform(df[num_vars])


# In[47]:


# target - change into classification problem
df['adjusted'] = df['adjusted'].str.replace('$','')
df['adjusted'] = df['adjusted'].str.replace(',','')
df['adjusted'] = df['adjusted'].astype(float)
df

# devide into three categories
df_descending=df.sort_values(by="adjusted" , ascending=False) 
cut1_index=round(len(df)/3)
print(cut1_index)
cut2_index=round(len(df)/3*2)
print(cut2_index)

cut1 = df_descending.loc[cut1_index,'adjusted']
print(cut1) 
cut2 = df_descending.loc[cut2_index,'adjusted']
print(cut2)

def adjusted_category(adjusted):
    if adjusted > cut1:
        return 2
    elif adjusted < cut2:
        return 0
    else:
        return 1
df['adjusted_category'] = df['adjusted'].map(lambda x: adjusted_category(x))
df = df.drop(columns=['adjusted'])


# In[49]:


df


# In[50]:


# Separate Feature and Target Matrix
x = df.drop('adjusted_category',axis = 1) 
y = df.adjusted_category

# split train and test set, test size = 0.3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 5)


# In[51]:


#importing algorithms
from sklearn.svm import SVC
from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#define three classfications by using LR, DTC, SVC
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = SVC()
eclf = VotingClassifier(estimators=[('lr', clf1), ('dtc', clf2),('svc',clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Decision Tree','SVC','Ensemble']):
        scores = cross_val_score(clf, x_train, y_train, cv = 5, scoring = 'accuracy')
        print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))

eclf.fit(x_train, y_train)
print("Accuracy: %0.4f" % (eclf.score(x_test, y_test)))


# In[52]:


# Hyperparameter Optimisation

import sklearn.svm as svm
import sklearn.model_selection as ms

model = svm.SVC(probability=True)
# Gridsearch
params = [
    {'kernel':['linear'],'C':[1,10,100,1000]},
    {'kernel':['poly'],'C':[1,10],'degree':[2,3]},
    {'kernel':['rbf'],'C':[1,10,100,1000], 
     'gamma':[10,1,0.1, 0.01, 0.001,0.0001]}]
model = ms.GridSearchCV(estimator=model, param_grid=params, cv=5)
model.fit(x_train, y_train)

print("Best parameters:",model.best_params_)
print("Best accuracy:",model.best_score_)
print("Best model:",model.best_estimator_)
print("Accuracy: %0.4f" % (model.score(x_test, y_test)))


# In[ ]:




