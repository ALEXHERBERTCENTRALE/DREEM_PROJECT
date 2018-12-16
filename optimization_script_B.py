# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:16:37 2018

@author: Alexandre Herbert
"""

from optimization_code import optimizeHyperParamSingleMethod
from feature_extraction import *
import h5py
import numpy as np
import os
os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')

""" Parameters to define for optimization """

X_path = 'balanced_data/X_train_balanced.h5'
y_path = 'balanced_data/X_train_balanced_labels.txt'

list_signals = [11]

methodOne = nbPikesOne
list_params_methodOne = [np.arange(1,51,2) , np.linspace(0,200,20)]

n_estimators=[100]  #[10, 100, 1000]
criterion=['gini']  #['gini', 'entropy']
max_depth=[None]
min_samples_split=[2]  #[2, 1000,500]
min_samples_leaf=[1]   #[1, 1000, 500] 
min_impurity_decrease=[0.0]

n_folds=5

""""""""""""""""""""""""""""""""""""""""""""

X = h5py.File(X_path, 'r')
y = np.array(objectFromFile(y_path))
list_params_tree = [n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease]

optimizeHyperParamSingleMethod(X, list_signals, methodOne, list_params_methodOne, list_params_tree, n_folds, y)