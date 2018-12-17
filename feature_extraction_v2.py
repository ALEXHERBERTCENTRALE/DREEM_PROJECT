import numpy as np
import pickle
import sys
# import os
# os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')
# from feature_extraction import *


def methodTestOneAdapt(list_freq = None, param = None, rep_dim_feature_per_signal = False):  # param useless
    # Pointless method to test extractFeatureAll
    if rep_dim_feature_per_signal:
        return 2
    
    return np.array([param[0],2])

## Global feature-extraction methods
    
def extractFeatureAllAdapt(h5file_freq , methodOne ,  list_bool_extract_signal , list_param_extract_signal):
    # Method giving the design matrix of h5file, given a certain method
    # list_bool_extract_signal[i] contains 1 iff the ith signal (data field) must have its feature extracted for this methodOne. Must be of length : len(key_list)
    # list_param_extract_signal[i] contains the list of parameters to be applied to the ith signal with the current method. Contains None if the feature should not be extracted. Must be of length : len(key_list)
    key_list = list(h5file_freq.keys())
    key_list_extract = list( key_list[i] for i in range(len(key_list)) if list_bool_extract_signal[i] )
    nb_samples = len(h5file_freq[key_list[0]])
    dim_feature_per_signal = methodOne(rep_dim_feature_per_signal = True)
    rep = np.zeros((nb_samples , len(key_list_extract)*dim_feature_per_signal))
    list_param_extract_signal_not_none = [param for param in list_param_extract_signal if param is not None]
    
    for k_id in range(len(key_list_extract)):
        k=key_list_extract[k_id]
        rep[: ,k_id*dim_feature_per_signal:(k_id+1)*dim_feature_per_signal ] =  np.array( list(methodOne(h5file_freq[k][i] , list_param_extract_signal_not_none[k_id]) for i in range(nb_samples)  ))
    return rep

def extractMultiFeatureAllAdapt(h5file_freq , list_methodOne , mat_bool_extract_signal , mat_param_extract_signal , save = False , name_save = None):
    # Returns the concatenation of design matrices for a list of methods
    # mat_bool_extract_signal[i] (ith row) contains list_bool_extract_signal for extractFeatureAll function. IE : mat_bool_extract_signal[i,j] is 1 iff for ith methodOne, jth signal must have its feature extracted. 
    # mat_bool_extract_signal must be of size : (nb of methodOnes , len(key_list)  )
    # mat_param_extract_signal[i] (ith row) contains list_param_extract_signal for extractFeatureAll function. IE : mat_param_extract_signal[i,j] is the list of parameters for ith methodOne and signal j (None if not to be extracted).
    # mat_param_extract_signal must be of size : (nb of methodOnes , len(key_list)  )
    key_list = list(h5file_freq.keys())
    nb_samples = len(h5file_freq[list(h5file_freq.keys())[0]])
    #sum_dim_feature_per_signal = sum( methodOne(rep_dim_feature_per_signal = True) for methodOne in list_methodOne )
    list_key_list_extract = list( list( key_list[j] for j in range(len(key_list)) if mat_bool_extract_signal[i,j] ) for i in range(len(list_methodOne)) )
    
    sum_weighted_dim_feature_per_signal = sum( list_methodOne[i](rep_dim_feature_per_signal = True)*len(list_key_list_extract[i]) for i in range(len(list_methodOne)) )
    
    rep = np.zeros((nb_samples , sum_weighted_dim_feature_per_signal ))
    
    c = 0
    i = 0
    
    # setup toolbar
    print("Progress...")
    sys.stdout.write("|"+("_" * len(list_methodOne)) + "_|\n")
    sys.stdout.flush()
    sys.stdout.write("|>")
    sys.stdout.flush()
    
    for methodOne in list_methodOne :
        
        # update the bar
        sys.stdout.write("\b")
        sys.stdout.write("=>")
        sys.stdout.flush()
        
        temp = len(list_key_list_extract[i])*methodOne(rep_dim_feature_per_signal = True)
        rep[:,c:c+temp] = extractFeatureAllAdapt(h5file_freq , methodOne , mat_bool_extract_signal[i] , mat_param_extract_signal[i] )
        i+=1
        c+=temp
    
    # close the bar
    sys.stdout.write("\b")
    sys.stdout.write("=|\n")
    
    if save:
        temp_var_file = open("design_matrix/" +name_save + '.txt','wb')
        pickle.dump(rep , temp_var_file)
        temp_var_file.close()
        
        #Use next 3 lines to read
        # temp_var_file = open(name_save + '.txt','rb')
        # rep = pickle.load(temp_var_file)
        # temp_var_file.close()
        
        #np.savetxt( name_save + '.txt' , rep , delimiter=',', fmt="%s")
    
    return rep



## to do some testing
# pseudo-h5 file with 3 keys (3 indicators), and 3 samples, of length 29 each
dico = {}
dico["cle1"] = [np.arange(1,30) , np.random.normal(10,5,29) , np.arange(2,31) , np.random.uniform(12,45,29) ]
dico["cle2"] = [np.arange(2,31), np.arange(1,30) , np.random.uniform(12,45,29) , np.random.normal(12,6,29)]
dico["cle3"] = [np.random.normal(42,1,29), np.random.uniform(12,45,29), np.arange(2,31),np.arange(10,39)]

mat_bool_test = np.array( [ [ 1,1,1 ] , 
                            [0,1,1] , 
                            [1,0,0] ] )
                            
mat_param_test = np.array( [ [ [4,0.5] , [10,1.1] , [2,1]],
                             [ None , [1] , [3] ],
                             [ [15] , None , None ] ] )
    
# print(extractMultiFeatureAllAdapt(dico , [nbPikesOne , methodTestOneAdapt ]  , mat_bool_test , mat_param_test[0:2]))
# print(extractMultiFeatureAllAdapt(dico , [nbPikesOne , methodTestOneAdapt , stdDeviationNbOne]  , mat_bool_test , mat_param_test))
