import h5py
import pickle
import numpy as np
import random as rd
# import os
# os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')
from feature_extraction import objectFromFile


    

def balanceData(h5file_list , write_name_list = ['X_balanced' , 'X_fft_balanced'], write_name_labels = 'balanced_labels' ,  nb_samples = None , unique = True , labels_path = 'data/train_y.txt' ):
    labels = list(objectFromFile(labels_path))
    
    # Creating the lists of indexes for each labels
    separate_indexes = [-1]*5
    for i,j in enumerate(labels):
        try:
            separate_indexes[j].append(i)
        except:
            separate_indexes[j] = [i]
    
    
    # Using the minimum of samples available in one class if nb_samples was not given
    if nb_samples is None:
        nb_samples = min( len(separate_indexes[j]) for j in range(len(separate_indexes)) )
    
    # Handling error if unique == True
    if unique and any( labels.count(i) < nb_samples for i in range(5) ):
        raise ValueError('You wish to extract too many unique samples. Please consider reducing the number of extracted samples, or using unique = False. The number of samples for each class are : {0}'.format([labels.count(i)  for i in range(5)]))
    
    
    
    keys = list(h5file_list[0].keys())
    dataset_size = len(h5file_list[0][keys[0]])
    
    new_labels = [0]*5*nb_samples
    
    X_balanced = [0]*len(write_name_list)
    for i in range(len(write_name_list)):
        X_balanced[i] = h5py.File("balanced_data/" + write_name_list[i] + '.h5' , 'a' )
    
    
    
    # Creating the lists of chosen indexes
    separate_indexes_chosen = [0]*5
    for i in range(5):
        separate_indexes_chosen[i] = np.random.choice( separate_indexes[i] , nb_samples , replace = not unique )
    
    # Creating new labels
    # Extracting chosen data
    for num_file in range(len(write_name_list)):
        for feature_id in range(len(keys)):
            feature=keys[feature_id]
            feature_chosen=[[]]*5*nb_samples
            c = 0
            for element_id in range(dataset_size):
                if any(element_id in separate_indexes_chosen[i] for i in range(5)):
                    # signals = h5file_list[i][feature]
                    feature_chosen[c] = h5file_list[num_file][feature][element_id]
                    if num_file==0:
                        new_labels[c] = labels[element_id]
                    c+=1
            X_balanced[num_file].create_dataset(name=feature, data=feature_chosen, dtype="float")
    
    # Saving new labels
    temp_var_file = open("balanced_data/" + write_name_labels + '.txt','wb')
    pickle.dump(new_labels , temp_var_file)
    temp_var_file.close()
    
    
    return X_balanced , new_labels
    
    
        
    
        
    