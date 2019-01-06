## Imports
import os
os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')

from feature_extraction import *
from feature_extraction_v2 import *
from ml_methods import myRandomForestClassifier
from cross_validation_learning import *

import numpy as np
import matplotlib.pyplot as plt
import h5py

## Parameters
create_new_design_matrix = True
create_new_prediction = True

name = 'all_of_them'

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

## Methods to be used
list_methods_time = [ distanceMinMaxOne , maxAmpOne , freqMinLimitAmpOne , nbPikesOne , indexMaxAmpOne , meanDiffNeighbOne , stdDeviationNbOne , meanOne , meanOfAbsOne , maxOfAbsOne , minOfAbsOne ]

list_methods_freq = [ distanceMinMaxOne , maxAmpOne , freqMinLimitAmpOne , nbPikesOne , indexMaxAmpOne , meanDiffNeighbOne , stdDeviationNbOne , meanOne , minOfAbsOne ]

## Temporal matrices

mat_bool_extract_signal_temp = np.array([  [0]*3 + [1]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [1]  ,
                                           [0]*3 + [1]*7 + [1]  ,
                                           [0]*3 + [0]*7 + [0]  ,
                                           [1]*3 + [1]*7 + [0]  ,
                                           [1]*3 + [1]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [0]  ,
                                           [1]*3 + [1]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [0]    ])

 # mat_bool_extract_signal_temp = np.array([  [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [1]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]  ,
 #                                           [0]*3 + [0]*7 + [0]    ])

                                           
mat_param_extract_signal_temp = np.array([  [[2]]*3 + [[5]]*7 + [[42]]  ,
                                           [[]]*3 + [[]]*7 + [[]]  ,
                                           [[0.44]]*3 + [[0.18]]*7 + [[0.57]]  ,
                                           [[18,0.2105]]*3 + [[2, 0.0526]]*7 + [[16,0.3367]]  ,
                                           [[11]]*3 + [[31]]*7 + [[18]]  ,
                                           [[1]]*3 + [[1]]*7 + [[1]]  ,
                                           [[42]]*3 + [[6]]*7 + [[44]]  ,
                                           [[]]*3 + [[]]*7 + [[]]  ,
                                           [[]]*3 + [[]]*7 + [[]]  ,
                                           [[]]*3 + [[]]*7 + [[]]  ,
                                           [[]]*3 + [[]]*7 + [[]]    ])

n,m = mat_param_extract_signal_temp.shape
for i in range(n):
    for j in range(m):
        if not mat_bool_extract_signal_temp[i,j]:
            mat_param_extract_signal_temp[i,j] = None

## Frequential matrices

mat_bool_extract_signal_freq = np.array([  [0]*3 + [1]*7 + [0]  ,
                                           [1]*3 + [1]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [1]  ,
                                           [0]*3 + [0]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [1]  ,
                                           [1]*3 + [1]*7 + [1]  ,
                                           [1]*3 + [1]*7 + [0]  ,
                                           [1]*3 + [1]*7 + [0]  ,
                                           [0]*3 + [0]*7 + [0]    ])

# mat_bool_extract_signal_freq = np.array([  [0]*3 + [0]*7 + [0]  ,
#                                            [0]*3 + [0]*7 + [0]  ,
#                                            [0]*3 + [0]*7 + [1]  ,
#                                            [0]*3 + [0]*7 + [0]  ,
#                                            [0]*3 + [0]*7 + [0]  ,
#                                            [0]*3 + [0]*7 + [0]  ,
#                                            [0]*3 + [0]*7 + [0]  ,
#                                            [0]*3 + [0]*7 + [0]  ,
#                                            [0]*3 + [0]*7 + [0]    ])
                                           
mat_param_extract_signal_freq = np.array([  [[2]]*3 + [[19]]*7 + [[16]]  ,
                                           [[]]*3 + [[]]*7 + [[]]  ,
                                           [[0.87]]*3 + [[0.09]]*7 + [[0.97]]  ,
                                           [[1,0.2105]]*3 + [[1, 0.0526]]*7 + [[7,0.1053]]  ,
                                           [[1]]*3 + [[17]]*7 + [[5]]  ,
                                           [[1]]*3 + [[1]]*7 + [[1]]  ,
                                           [[50]]*3 + [[22]]*7 + [[20]]  ,
                                           [[]]*3 + [[]]*7 + [[]]  ,
                                           [[]]*3 + [[]]*7 + [[]]    ])

nf,mf = mat_param_extract_signal_freq.shape
for i in range(nf):
    for j in range(mf):
        if not mat_bool_extract_signal_freq[i,j]:
            mat_param_extract_signal_freq[i,j] = None

## Importing data
X_train_balanced = h5py.File('balanced_data/X_train_balanced.h5' , 'r' )

X_train_fft_balanced = h5py.File('balanced_data/X_train_fft_balanced.h5' , 'r' )


X_test = h5py.File('data/X_test.h5')

X_test_fft = h5py.File('data/X_test_fft.h5')

## Creating the design matrices
if create_new_design_matrix:
    matrix_temp = extractMultiFeatureAllAdapt(X_train_balanced , list_methods_time , mat_bool_extract_signal_temp , mat_param_extract_signal_temp , save = True , name_save = "big_matrix_" + name + "_temp")
    
    matrix_freq = extractMultiFeatureAllAdapt(X_train_fft_balanced , list_methods_freq , mat_bool_extract_signal_freq , mat_param_extract_signal_freq , save = True , name_save = "big_matrix_" + name + "_freq")
    
    matrix = concatenateDesignMatrices( matrix_temp , matrix_freq , name_save = "big_matrix_" + name )

## Importing from previous calculations
else:
    matrix = objectFromFile("design_matrix/big_matrix_" + name + ".txt")

## Creating design matrix for prediction
if create_new_prediction:
    prediction_matrix_temp = extractMultiFeatureAllAdapt(X_test , list_methods_time , mat_bool_extract_signal_temp , mat_param_extract_signal_temp , save = True , name_save = "big_prediction_matrix_" + name + "_temp")
    
    prediction_matrix_freq = extractMultiFeatureAllAdapt(X_test_fft , list_methods_freq , mat_bool_extract_signal_freq , mat_param_extract_signal_freq , save = True , name_save = "big_prediction_matrix_" + name + "_freq")
    
    
    prediction_matrix = concatenateDesignMatrices( prediction_matrix_temp , prediction_matrix_freq , name_save = "big_prediction_matrix_" + name )

else:
    prediction_matrix = objectFromFile("design_matrix/big_prediction_matrix_" + name + ".txt" )

##☻ Killing the game
matrix = objectFromFile('design_matrix/Xtrain.txt')
prediction_matrix = objectFromFile('design_matrix/Xtest.txt')
labels_path = 'data/train_y.txt'

## Learning


mat_theta , mat_ypred , mat_yprob  = learn( matrix , mlMethod , list_params_tree , n_folds , labels_path = labels_path)

clf , scaler = learnEverything( matrix , mlMethod , list_params_tree_everything , labels_path = labels_path )

## Visualizing

visualizeResults( mat_theta , mat_ypred , mat_yprob , 0 , "" , [0,0,0,0,0] , labels_path = labels_path )

## Predicting

predict( prediction_matrix , clf , scaler = scaler , save = True , name_save =  name)