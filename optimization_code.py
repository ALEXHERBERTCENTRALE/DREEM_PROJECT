# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:28:15 2018

@author: Alexandre Herbert
"""

from feature_extraction import extractFeatureAll, objectFromFile
from ml_methods import *
from sklearn import model_selection
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

def cross_validate(design_matrix, labels, classifier, n_folds):
    
    #cv_folds = cross_validation.StratifiedKFold(labels, n_folds, shuffle=True)
    skf = model_selection.StratifiedKFold(n_folds, shuffle=True)
    cv_folds= skf.split(design_matrix, labels)
    
    pred = np.zeros(labels.shape)

    for train_folds, test_fold in cv_folds:
        
        # Restrict data to train/test folds
        Xtrain = design_matrix[train_folds, :]
        ytrain = labels[train_folds]
        Xtest = design_matrix[test_fold, :]

        # Scale data
        scaler = sklearn.preprocessing.StandardScaler() # create scaler
        Xtrain = scaler.fit_transform(Xtrain) # fit the scaler to the training data and transform training data
        Xtest = scaler.transform(Xtest) # transform test data
        
        # Fit classifier
        classifier.fit(Xtrain, ytrain)
            
        ytest_pred = classifier.predict(Xtest)
        pred[test_fold] = ytest_pred
        
    return pred

def learnOneSetOfParams(design_matrix, list_param, n_folds , labels):
    clf = myRandomForestClassifier(*list_param)
    ypred = cross_validate(design_matrix, labels, clf, n_folds)
    return ypred

def computeMetrics(ypred, labels):
    f1_scores = sklearn.metrics.f1_score(labels, ypred, average='macro')
    return f1_scores

def listAllCombinations(list_params):
    combinations = list([])

    if list_params==[]:
        return [None]
    if len(list_params)==1:
        return list_params[0]
    
    values_for_first_param = list_params[0]
    combinations_without_first_param = listAllCombinations(list_params[1:])
    
    for value_for_first in values_for_first_param:
        value_for_first = [value_for_first]
        
        for comb in combinations_without_first_param:
            while str(type(comb)) == "<class 'list'>" and len(comb)==1:
                comb = comb[0]
            if str(type(comb)) != "<class 'list'>":
                comb=[comb]

            combinations = combinations + [value_for_first + comb]

    return combinations

def prepareAllCombinations(list_params):
    combinations = listAllCombinations(list_params)
    nb_params = len(list_params)
    nb_combinations = len(combinations)
    if nb_params==1:
        combinations = [[combinations[i]] for i in range(nb_combinations)]
#    else:
#        combinations = [[[combinations[i][j]] for j in range(nb_params)] for i in range(nb_combinations)]
    return combinations

#a=["1","2"]
#b=[4]
#c=[7]
#splitter=['random']
#min_samples_leaf=[500, 1000]
#max_leaf_nodes=[6]
#class_weight= [{0:2,1:7,2:1,3:2,4:1}]
#list_params_tree = [splitter, min_samples_leaf, max_leaf_nodes, class_weight]
#print(prepareAllCombinations(list_params_tree))
    
def printConfusionMatrix(ytrue,  ypred):
    conf_mat = sklearn.metrics.confusion_matrix(ytrue,  ypred, labels=None, sample_weight=None).T
    print("Matrice de confusion : \n")
    print("\t\tTrue 0  True 1  True 2  True 3  True 4")
    for p in range(5):
        row_to_print="Predicted " +str(p)
        for t in range(5):
            row_to_print+="\t"+  str(conf_mat[p,t])
        print(row_to_print)
    print("")

def optimizeHyperParamSingleMethod(h5file, list_signals, methodOne, list_params_methodOne, list_params_tree, n_folds, ytrue):
    
    list_bool_extract_signal = np.zeros(11)
    for i in list_signals:
        list_bool_extract_signal[i-1]=1

    print('\n----------------------------------------------------')
    print('--- Optimisation ' + methodOne.__name__ + ' : signaux ' + str(list_signals))
    print('----------------------------------------------------\n')

    mat_param_methodOne = prepareAllCombinations(list_params_methodOne)
    mat_param_tree = prepareAllCombinations(list_params_tree)
    
    nb_combinaisons_methodOne = len(mat_param_methodOne)
    nb_combinaisons_tree = len(mat_param_tree)

    best_f1_score = 0
    best_params_methodOne = []
    best_params_tree = []
    f1_scores = []

    for i in range(nb_combinaisons_methodOne):
        params_methodOne = mat_param_methodOne[i]
        if params_methodOne == None:
            print('------------   Extraction   ------------')
            X = extractFeatureAll(h5file , methodOne , [], list_bool_extract_signal)
        else:
            print('------------   Extraction : params ' + str(params_methodOne) + '   ------------')
            X = extractFeatureAll(h5file , methodOne , params_methodOne, list_bool_extract_signal)
        
        for k in range(nb_combinaisons_tree):
            params_tree = mat_param_tree[k]
            print('')
            print('------------   Simulation ' + str(i+1) + '-' + str(k+1) + ' (' + str(nb_combinaisons_methodOne) + '-' + str(nb_combinaisons_tree) + ')   ------------')
            print('Params_methodOne : ' + str(params_methodOne))
            print('Params_tree : ' + str(params_tree))
            
            ypred = learnOneSetOfParams(X, params_tree, n_folds, ytrue)
            f1_score = computeMetrics(ypred, ytrue)
            
            print('F1-score : ' + str(f1_score))
            printConfusionMatrix(ytrue,  ypred)
            
            f1_scores += [f1_score]
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_params_methodOne = params_methodOne
                best_params_tree = params_tree

    print('-----   Final results   -----')    
    print('Best Params_methodOne :', best_params_methodOne)
    print('Best Params_tree :', best_params_tree)
    print('Best F1-score :', best_f1_score)
    
    plt.plot(f1_scores)
       
    return best_f1_score, best_params_methodOne, best_params_tree