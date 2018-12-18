import h5py
import pickle
import os
os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')
from feature_extraction import *
from balance_dataset import balanceData

X_train = h5py.File('data/X_train.h5','r')
X_train_fft = h5py.File('data/X_train_fft.h5','r')

balanceData([X_train , X_train_fft] , write_name_list = ['X_train_balanced2' , 'X_train_fft_balanced2'] , write_name_labels = 'X_train_balanced2_labels' ,  nb_samples = None , unique = True , labels_path = 'data/train_y.txt' )