# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:16:37 2018

@author: Alexandre Herbert
"""

from optimization_code import optimizeHyperParamSingleMethod
from feature_extraction import *
import h5py

""" Parameters to define for optimization """

X_path = 'balanced_data/X_train_fft_balanced.h5'
y_path = 'balanced_data/X_train_fft_balanced_labels.txt'

num_signal = 2

methodOne = minOfAbsOne
list_params_methodOne = []

splitter=['best']
min_samples_leaf=[500, 1000, 800]
max_leaf_nodes=[100,5]
class_weight= [{0:1,1:1,2:1,3:1,4:1}]

n_folds=10

""""""""""""""""""""""""""""""""""""""""""""

X = h5py.File(X_path, 'r')
y = np.array(objectFromFile(y_path))
list_params_tree = [splitter, min_samples_leaf, max_leaf_nodes, class_weight]

optimizeHyperParamSingleMethod(X, num_signal, methodOne, list_params_methodOne, list_params_tree, n_folds, y)