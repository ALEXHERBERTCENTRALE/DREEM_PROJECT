# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:22:15 2018

@author: alexa
"""

from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree
from sklearn import cluster
from sklearn import svm

def myKNeighborsClassifier(n_neighbors):
    return neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    
def myLogisticRegression(C, max_iter):
    return linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=C, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=max_iter, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None)
    
def myDecisionTreeClassifier(splitter, min_samples_leaf, max_leaf_nodes, class_weight):
    return tree.DecisionTreeClassifier(criterion='gini', splitter=splitter, max_depth=None, min_samples_split=2, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=class_weight, presort=False)
    
def myKMeans(n_clusters):
    return cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
    
def myLinearSVC():
    return svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)