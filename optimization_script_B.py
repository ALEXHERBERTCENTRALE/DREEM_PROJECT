import os
os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')

from optimization_code import optimizeHyperParamSingleMethod
from feature_extraction import *
import h5py
import numpy as np
import matplotlib.pyplot as plt


## Parameters to define for optimization 

X_path = 'balanced_data/X_train_fft_balanced.h5'
y_path = 'balanced_data/X_train_fft_balanced_labels.txt'

list_signals = [i for i in range(4,11)]

methodOne = indexMaxAmpOne
list_params_methodOne = [np.arange(1,51,2)]# , np.linspace(0,1,20)]

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
## Execution
optimizeHyperParamSingleMethod(X, list_signals, methodOne, list_params_methodOne, list_params_tree, n_folds, y)