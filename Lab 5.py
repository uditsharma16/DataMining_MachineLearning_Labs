# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:03:57 2020

@author: rpear
"""


# coding: utf-8

from IPython.display import Image


# # Streamlining workflows with pipelines

# ...

# ## Loading the Breast Cancer Wisconsin dataset

import pandas as pd

df = pd.read_csv('http://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

print('header', df.head())
print(df.shape)

from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)
print(le.transform(['M', 'B']))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
#
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

pipe_dt = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=1))
param_grid=[{'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
gs = GridSearchCV(estimator=pipe_dt,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    gs.fit(X_train[train], y_train[train])
    score = gs.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# # Looking at different performance evaluation metrics
# ...
# ## Reading a confusion matrix


