def preprocessingFrequencyWithLimitedAmplitude(amp_lim):
    X_train_fft = h5py.File('X_train_fft.h5','r')
    X_test_fft = h5py.File('X_test_fft.h5','r')
    keys = list(X_train_fft.keys())
    
    train_dataset_size = len(X_train_fft[keys[0]])
    test_dataset_size = len(X_test_fft[keys[0]])
    
    X_train_preprocessed = np.zeros((train_dataset_size,len(keys)))
    X_test_preprocessed = np.zeros((test_dataset_size,len(keys)))
    
    for feature_id in range(len(keys)):
        feature = keys[feature_id]
        train_data = X_train_fft[feature]
        for element_id in range(len(train_data)):
            element = train_data[element_id]
            
            # normalization
            maxim = max(element)
            element = [a/maxim for a in element]
            
            # preprocessing
            X_train_preprocessed[element_id, feature_id] = frequencyWithLimitedAmplitude(element, amp_lim)
            print(feature, " - ", element_id)
            
        test_data = X_test_fft[feature]
        for element_id in range(len(test_data)):
            element = test_data[element_id]
            
            # normalization
            maxim = max(element)
            element = [a/maxim for a in element]
            
            # preprocessing
            X_test_preprocessed[element_id, feature_id] = frequencyWithLimitedAmplitude(element, amp_lim)
            print(feature, " - ", element_id)
            
    return X_train_preprocessed, X_test_preprocessed

def preprocessingPikes(amp_lim, interval_width):
    X_train_fft = h5py.File('X_train_fft.h5','r')
    X_test_fft = h5py.File('X_test_fft.h5','r')
    keys = list(X_train_fft.keys())
    
    train_dataset_size = len(X_train_fft[keys[0]])
    test_dataset_size = len(X_test_fft[keys[0]])
    
    X_train_preprocessed = np.zeros((train_dataset_size,len(keys)))
    X_test_preprocessed = np.zeros((test_dataset_size,len(keys)))
    
    for feature_id in range(len(keys)):
        feature = keys[feature_id]
        train_data = X_train_fft[feature]
        for element_id in range(len(train_data)):
            element = train_data[element_id]
            
            # normalization
            maxim = max(element)
            element = [a/maxim for a in element]
            
            # preprocessing
            X_train_preprocessed[element_id, feature_id] = pikes(element, interval_width, amp_lim)
            print(feature, " - ", element_id)
            
        test_data = X_test_fft[feature]
        for element_id in range(len(test_data)):
            element = test_data[element_id]
            
             # normalization
            maxim = max(element)
            element = [a/maxim for a in element]
            
            # preprocessing
            X_test_preprocessed[element_id, feature_id] = pikes(element, interval_width, amp_lim)
            print(feature, " - ", element_id)
            
    return X_train_preprocessed, X_test_preprocessed

def meanDiffNeighb(h5file_freq):  #h5file_freq is X_train or X_test, key_list is keys
    #X_train_fft = h5py.File('X_train_fft.h5','a')
    rep = [0]*len(h5file_freq)
    key_list = list(h5file_freq.keys())
    for k_id in range(len(key_list)):
        k=key_list[k_id]
        rep[k] = list(mean_diff_neighb_one(h5file_freq[k][i]) for i in range(len(h5file_freq[k])))
    return rep


def std_deviation_nb(h5file_freq , n_nb):
    rep = [0]*len(h5file_freq)
    key_list = list(h5file_freq.keys())
    for k_id in range(len(key_list)):
        k=key_list[k_id]
        rep[k] = list(std_deviation_nb_one(h5file_freq[k][i] , n_nb) for i in range(len(h5file_freq[k])))
    return rep

def upper_right(h5file_freq , th_amp , th_freq):
    rep = [0]*len(h5file_freq)
    key_list = list(h5file_freq.keys())
    for k_id in range(len(key_list)):
        k=key_list[k_id]
        rep[k] = list(upper_right_one(h5file_freq[k][i] , th_amp , th_freq) for i in range(len(h5file_freq[k])))
    return rep

def preprocessingMaximum():
    X_train_fft = h5py.File('X_train_fft.h5','r')
    X_test_fft = h5py.File('X_test_fft.h5','r')
    keys = list(X_train_fft.keys())
    
    train_dataset_size = len(X_train_fft[keys[0]])
    test_dataset_size = len(X_test_fft[keys[0]])
    
    X_train_preprocessed = np.zeros((train_dataset_size,len(keys)))
    X_test_preprocessed = np.zeros((test_dataset_size,len(keys)))

    for feature_id in range(len(keys)):
        feature = keys[feature_id]
        train_data = X_train_fft[feature]
        for element_id in range(train_dataset_size):
            element = train_data[element_id]
            
            X_train_preprocessed[element_id,feature_id]=max(element)
            print(feature, " - ", element_id)

    for feature_id in range(len(keys)):
        feature = keys[feature_id]
        test_data = X_test_fft[feature]
        for element_id in range(test_dataset_size):
            element = test_data[element_id]
            
            X_test_preprocessed[element_id,feature_id]=max(element)
            print(feature, " - ", element_id)
            