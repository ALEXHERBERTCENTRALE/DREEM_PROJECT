# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:29:36 2018

@author: alexa
"""

import numpy as np
import h5py

# Chargement des données

X_train = h5py.File('train.h5', 'r')
X_test = h5py.File('test.h5', 'r')
y = np.loadtxt('train_y.csv',  delimiter=',', skiprows=1, usecols=range(1, 2)).astype('int')

#print(list(X_train.keys()))
#print(list(X_test.keys()))
#print(y)


accelerometer_x_train = X_train['accelerometer_x']
accelerometer_y_train = X_train['accelerometer_y']
accelerometer_z_train = X_train['accelerometer_z']
eeg_1_train = X_train['eeg_1']
eeg_2_train = X_train['eeg_2']
eeg_3_train = X_train['eeg_3']
eeg_4_train = X_train['eeg_4']
eeg_5_train = X_train['eeg_5']
eeg_6_train = X_train['eeg_6']
eeg_7_train = X_train['eeg_7']
pulse_oximeter_infrared_train = X_train['pulse_oximeter_infrared']

accelerometer_x_test = X_test['accelerometer_x']
accelerometer_y_test = X_test['accelerometer_y']
accelerometer_z_test = X_test['accelerometer_z']
eeg_1_test = X_test['eeg_1']
eeg_2_test = X_test['eeg_2']
eeg_3_test = X_test['eeg_3']
eeg_4_test = X_test['eeg_4']
eeg_5_test = X_test['eeg_5']
eeg_6_test = X_test['eeg_6']
eeg_7_test = X_test['eeg_7']
pulse_oximeter_infrared_test = X_test['pulse_oximeter_infrared']

keys=list(X_train.keys())
train_dataset_size=len(accelerometer_x_train)
test_dataset_size=len(accelerometer_x_test)