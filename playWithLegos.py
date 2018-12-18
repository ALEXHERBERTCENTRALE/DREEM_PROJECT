# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:49:08 2018

@author: Alexandre Herbert
"""

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

selected_freq_matrices = np.array([[0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0]  ,
                                  [0,0,0] ])

nb_features=

def assembleElemDesignMatrices():
    Xtrain = np.array([])
    Xtest = np.array([])
    
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
            Xtrain_elem_path = "design_matrix/elem/Xtrain_time_" + methodOne_name + "_" + type_name
            Xtest_elem_path = "design_matrix/elem/Xtest_time_" + methodOne_name + "_" + type_name
            if selected_temp_matrices[id_feat][signal_type]==1:
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
            Xtrain_elem_path = "design_matrix/elem/Xtrain_fft_" + methodOne_name + "_" + type_name
            Xtest_elem_path = "design_matrix/elem/Xtest_fft_" + methodOne_name + "_" + type_name
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