import h5py
import pickle
import os
os.chdir('C:/Users/tbapt/Desktop/Documents/Ecole/3A/Machine_learning/DREEM_PROJECT')
from feature_extraction import *
from balance_dataset import balanceData

X_train = h5py.File('data/train.h5','r')
X_train_fft = h5py.File('data/X_train_fft.h5','r')

balanceData(X_train , 'X_train_balanced')
balanceData(X_train_fft , 'X_train_fft_balanced' )