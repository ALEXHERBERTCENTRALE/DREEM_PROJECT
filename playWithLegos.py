# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:49:08 2018

@author: Alexandre Herbert
"""
import os
os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')
import numpy as np
import sys
import pickle
from feature_extraction import objectFromFile
import h5py

h5_train = h5py.File('data/train.h5' , 'r' )
nb_train_samples = len(h5_train[list(h5_train.keys())[0]]) 
h5_test = h5py.File('data/X_test.h5')
nb_test_samples = len(h5_test[list(h5_test.keys())[0]]) 

list_methods_time = [ "distanceMinMaxOne" , "maxAmpOne" , "freqMinLimitAmpOne" , "nbPikesOne" , "indexMaxAmpOne" , "meanDiffNeighbOne" , "stdDeviationNbOne" , "meanOne" , "meanOfAbsOne" , "maxOfAbsOne" , "minOfAbsOne" ]
list_methods_freq = [ "distanceMinMaxOne" , "maxAmpOne" , "freqMinLimitAmpOne" , "nbPikesOne" , "indexMaxAmpOne" , "meanDiffNeighbOne" , "stdDeviationNbOne" , "meanOne" , "minOfAbsOne" ]
types_names = ["acc", "eeg", "oxy"]

selected_temp_matrices = np.array([[0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0]  ,
                              [0,0,0] ])

selected_temp_matrices = np.ones((11,3) , dtype = int)

selected_freq_matrices = np.array([[0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0] ])

selected_freq_matrices = np.ones((9,3) , dtype = int)

nb_temp_features = len(selected_temp_matrices)
nb_freq_features = len(selected_freq_matrices)

def assembleElemDesignMatrices():
    
    Xtrain = np.zeros((nb_train_samples,0))
    Xtest = np.zeros((nb_test_samples,0))
    
    print("Assembling temp matrices...")
    sys.stdout.write("|"+("_" * nb_temp_features*3) + "_|\n")
    sys.stdout.flush()
    sys.stdout.write("|>")
    sys.stdout.flush()
    
    # Assemble all temp matrices
    for id_feat in range(nb_temp_features):
        methodOne_name = list_methods_time[id_feat]
        for signal_type in range(3):
            type_name = types_names[signal_type]
            Xtrain_elem_path = "design_matrix/elem/Xtrain_time_" + methodOne_name + "_" + type_name + "_all.txt"
            Xtest_elem_path = "design_matrix/elem/Xtest_time_" + methodOne_name + "_" + type_name + ".txt"
            if selected_temp_matrices[id_feat][signal_type]==1:
                #print(Xtrain.shape , objectFromFile(Xtrain_elem_path).shape)
                Xtrain = np.concatenate( (Xtrain, objectFromFile(Xtrain_elem_path)) , axis = 1)
                Xtest = np.concatenate( (Xtest, objectFromFile(Xtest_elem_path)) , axis = 1)
                
            # update the bar
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()  
            
    # close the bar
    sys.stdout.write("\b")
    sys.stdout.write("=|\n")
    print("Assembling freq matrices...")
    sys.stdout.write("|"+("_" * nb_freq_features*3) + "_|\n")
    sys.stdout.flush()
    sys.stdout.write("|>")
    sys.stdout.flush()
    
    # Assemble all freq matrices
    for id_feat in range(nb_freq_features):
        methodOne_name = list_methods_freq[id_feat]
        for signal_type in range(3):
            type_name = types_names[signal_type]
            Xtrain_elem_path = "design_matrix/elem/Xtrain_fft_" + methodOne_name + "_" + type_name + "_all.txt"
            Xtest_elem_path = "design_matrix/elem/Xtest_fft_" + methodOne_name + "_" + type_name + ".txt"
            if selected_freq_matrices[id_feat][signal_type]==1:
                Xtrain = np.concatenate( (Xtrain, objectFromFile(Xtrain_elem_path)) , axis = 1)
                Xtest = np.concatenate( (Xtest, objectFromFile(Xtest_elem_path)) , axis = 1)
                
            # update the bar
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()  
            
    # close the bar
    sys.stdout.write("\b")
    sys.stdout.write("=|\n")
    
    temp_var_file = open('design_matrix/Xtrain.txt','wb')
    pickle.dump(Xtrain , temp_var_file)
    temp_var_file.close()
    
    temp_var_file = open('design_matrix/Xtest.txt','wb')
    pickle.dump(Xtest , temp_var_file)
    temp_var_file.close()
    
    print("Matrices saved")