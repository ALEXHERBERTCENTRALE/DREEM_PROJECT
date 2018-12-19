import os
os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')
import pickle
from feature_extraction import *
import sys
import h5py

list_methods_time = [ distanceMinMaxOne , maxAmpOne , freqMinLimitAmpOne , nbPikesOne , indexMaxAmpOne , meanDiffNeighbOne , stdDeviationNbOne , meanOne , meanOfAbsOne , maxOfAbsOne , minOfAbsOne ]
list_methods_freq = [ distanceMinMaxOne , maxAmpOne , freqMinLimitAmpOne , nbPikesOne , indexMaxAmpOne , meanDiffNeighbOne , stdDeviationNbOne , meanOne , minOfAbsOne ]

nb_temp_features = len(list_methods_time)
nb_freq_features = len(list_methods_freq)

#mat_bool_extract_signal_temp = np.array([  [0,1,0]  ,
#                                           [0,0,0]  ,
#                                           [0,0,1]  ,
#                                           [0,1,1]  ,
#                                           [0,0,0]  ,
#                                           [1,1,0]  ,
#                                           [1,1,0]  ,
#                                           [0,0,0]  ,
#                                           [1,1,0]  ,
#                                           [0,0,0]  ,
#                                           [0,0,0] ])

mat_bool_extract_signal_temp = np.array([  [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,1]  ,
                                           [0,1,1]  ,
                                           [0,0,0]  ,
                                           [1,1,0]  ,
                                           [1,1,0]  ,
                                           [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,0] ])

mat_param_extract_signal_temp = np.array([  [[2],[5],[42]]  ,
                                           [[],[],[]]  ,
                                           [[0.44],[0.18],[0.57]]  ,
                                           [[18,0.2105],[2, 0.0526],[16,0.3367]]  ,
                                           [[11],[31],[18]]  ,
                                           [[1],[1],[1]]  ,
                                           [[42],[6],[44]]  ,
                                           [[],[],[]]  ,
                                           [[],[],[]]  ,
                                           [[],[],[]]  ,
                                           [[],[],[]]    ])

#mat_bool_extract_signal_freq = np.array([  [0,1,0]  ,
#                                           [1,1,0]  ,
#                                           [0,0,1]  ,
#                                           [0,0,0]  ,
#                                           [0,0,1]  ,
#                                           [1,1,1]  ,
#                                           [1,1,0]  ,
#                                           [1,1,0]  ,
#                                           [0,0,0]    ])

mat_bool_extract_signal_freq = np.array([  [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,1]  ,
                                           [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,0]  ,
                                           [0,0,0]    ])

mat_param_extract_signal_freq = np.array([  [[2],[19],[16]]  ,
                                           [[],[],[]]  ,
                                           [[0.87],[0.09],[0.97]]  ,
                                           [[1,0.2105],[1, 0.0526],[7,0.1053]]  ,
                                           [[1],[17],[5]]  ,
                                           [[1],[1],[1]]  ,
                                           [[50],[22],[20]]  ,
                                           [[],[],[]]  ,
                                           [[],[],[]]    ])

signals_per_type = [[1,1,1,0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,1,1,1,1,0],
                    [0,0,0,0,0,0,0,0,0,0,1]]
types_names = ["acc", "eeg", "oxy"]

X_train_balanced = h5py.File('balanced_data/X_train_balanced.h5' , 'r' )
X_train_fft_balanced = h5py.File('balanced_data/X_train_fft_balanced.h5' , 'r' )

X_test = h5py.File('data/X_test.h5')
X_test_fft = h5py.File('data/X_test_fft.h5')

def buildAndSaveMatrix(h5file_freq, methodOne, param, list_bool_extract_signal, name_save):
    rep = extractFeatureAll(h5file_freq , methodOne , param , list_bool_extract_signal)
    temp_var_file = open("design_matrix/elem/" + name_save + '.txt','wb')
    pickle.dump(rep , temp_var_file)
    temp_var_file.close()
    
def buildAllElemDesignMatrices():
    
    print("Building temp matrices...")
    sys.stdout.write("|"+("_" * nb_temp_features*3) + "_|\n")
    sys.stdout.flush()
    sys.stdout.write("|>")
    sys.stdout.flush()
    
    # Build all temp matrices
    for id_feat in range(nb_temp_features):
        methodOne = list_methods_time[id_feat]
        for signal_type in range(3):
            param = mat_param_extract_signal_temp[id_feat][signal_type]
            signals = signals_per_type[signal_type]
            if mat_bool_extract_signal_temp[id_feat][signal_type]==1:
                buildAndSaveMatrix(X_train_balanced, methodOne, param , signals, "Xtrain_time_" +methodOne.__name__+ "_" + types_names[signal_type])
                buildAndSaveMatrix(X_test, methodOne, param , signals, "Xtest_time_" +methodOne.__name__+ "_" + types_names[signal_type])
             
            # update the bar
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()  
            
    # close the bar
    sys.stdout.write("\b")
    sys.stdout.write("=|\n")
    print("Building freq matrices...")
    sys.stdout.write("|"+("_" * nb_freq_features*3) + "_|\n")
    sys.stdout.flush()
    sys.stdout.write("|>")
    sys.stdout.flush()
            
    # Build all freq matrices
    for id_feat in range(nb_freq_features):
        methodOne = list_methods_freq[id_feat]
        for signal_type in range(3):
            param = mat_param_extract_signal_freq[id_feat][signal_type]
            signals = signals_per_type[signal_type]
            if mat_bool_extract_signal_freq[id_feat][signal_type]==1:
                buildAndSaveMatrix(X_train_fft_balanced, methodOne, param , signals, "Xtrain_fft_" +methodOne.__name__+ "_" + types_names[signal_type])
                buildAndSaveMatrix(X_test_fft, methodOne, param , signals, "Xtest_fft_" +methodOne.__name__+ "_" + types_names[signal_type])
                
            # update the bar
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()  
    
    # close the bar
    sys.stdout.write("\b")
    sys.stdout.write("=|\n")