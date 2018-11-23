import h5py
import numpy as np

def buildFFTDataset():
    X_train = h5py.File('train.h5','r').copy()
    X_test = h5py.File('test.h5','r').copy()
    keys = list(X_train.keys())

    X_train_fft = h5py.File('X_train_fft.h5','a')
    X_test_fft = h5py.File('X_test_fft.h5','a')
    
    train_dataset_size = len(X_train_fft[keys[0]])
    test_dataset_size = len(X_test_fft[keys[0]])

    for feature_id in range(len(keys)):
        feature=keys[feature_id]
        signals = X_train[feature]
        feature_fft=[[]]*train_dataset_size
        for element_id in range(train_dataset_size):
            element = signals[element_id]
            spectre = np.absolute(np.fft.fft(element))
            spectre = spectre[0:len(spectre)//2]
            feature_fft[element_id]=spectre
        X_train_fft.create_dataset(name=feature, data=feature_fft, dtype="float")

    for feature_id in range(len(keys)):
        feature=keys[feature_id]
        signals = X_test[feature]
        feature_fft=[[]]*test_dataset_size
        for element_id in range(test_dataset_size):
            element = signals[element_id]
            spectre = np.absolute(np.fft.fft(element))
            spectre = spectre[0:len(spectre)//2]
            feature_fft[element_id]=spectre
        X_test_fft.create_dataset(name=feature, data=feature_fft, dtype="float")