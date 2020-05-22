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

from sklearn.metrics import confusion_matrix
#
pipe_dt.fit(X_train, y_train)
y_pred = pipe_dt.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
#

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
#
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig('Images/05_02.png', dpi=300)
plt.show()
#
#
#
from sklearn.metrics import precision_score, recall_score, f1_score
#
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
#

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
#from scipy import interp ##no such library
#from scipy import interpolate
from numpy import interp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
                        LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000,
                                           random_state=1,C=100.0))
X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
#
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
#
plt.plot([0, 1],[0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
#
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
#
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('Images/05_1.png', dpi=300)
plt.show()

from sklearn.neural_network import MLPClassifier

pipe_mlp = make_pipeline(StandardScaler(), MLPClassifier()
                         )

import warnings

warnings.filterwarnings("ignore")

# param_grid=[{'mlpclassifier__hidden_layer_sizes': [(20,),(19,1),(18,2)]}]
param_grid = [{'mlpclassifier__n_iter_no_change': [1, 2, 3, 4, 5],
               'mlpclassifier__hidden_layer_sizes': [(20,), (19, 1), (18, 2)]}]
#
gs = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = []
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
for k, (train, test) in enumerate(kfold):
    gs.fit(X_train[train], y_train[train])
    score = gs.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k + 1, np.bincount(y_train[train]), score))
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


