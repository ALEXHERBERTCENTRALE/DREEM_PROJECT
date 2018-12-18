# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:42:31 2018

@author: Alexandre Herbert
"""

from feature_extraction import *
from cross_validation_learning import *
from ml_methods import myRandomForestClassifier

Xtrain = objectFromFile("design_matrix/Xtrain.txt")
Xtest = objectFromFile("design_matrix/Xtest.txt")

mlMethod = myRandomForestClassifier

labels_path = 'balanced_data/X_train_balanced_labels.txt'

n_estimators=[100]  #[10, 100, 1000]
criterion=['gini']  #['gini', 'entropy']
max_depth=[None]
min_samples_split=[2]  #[2, 1000,500]
min_samples_leaf=[1]   #[1, 1000, 500] 
min_impurity_decrease=[0.0]

n_folds=5

list_params_tree = [n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease]

list_params_tree_everything = [*n_estimators, *criterion, *max_depth, *min_samples_split, *min_samples_leaf, *min_impurity_decrease]

## Learning
mat_theta , mat_ypred , mat_yprob  = learn( Xtrain , mlMethod , list_params_tree , n_folds , labels_path = labels_path)
clf , scaler = learnEverything( Xtrain , mlMethod , list_params_tree_everything , labels_path = labels_path )

## Visualizing

visualizeResults( mat_theta , mat_ypred , mat_yprob , 0 , "" , [0,0,0,0,0] , labels_path = labels_path )

## Predicting

predict( Xtest , clf , scaler = scaler , save = True , name_save = "ypred")