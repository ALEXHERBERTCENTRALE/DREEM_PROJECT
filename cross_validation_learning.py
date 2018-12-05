import numpy as np
import sklearn
#from sklearn import neighbors
#from sklearn import cross_validation
from sklearn import model_selection
from sklearn import cluster
from sklearn import svm
from functools import reduce
import operator
import matplotlib.pyplot as plt
import csv

def cross_validate(design_matrix, labels, classifier, n_folds):
    """ Perform a cross-validation and returns the predictions.
    
    Parameters:
    -----------
    design_matrix: (n_samples, n_features) np.array
        Design matrix for the experiment.
    labels: (n_samples, ) np.array
        Vector of labels.
    classifier:  sklearn classifier object
        Classifier instance; must have the following methods:
        - fit(X, y) to train the classifier on the data X, y
        - predict_proba(X) to apply the trained classifier to the data X and return probability estimates 
    cv_folds: sklearn cross-validation object
        Cross-validation iterator.
        
    Return:
    -------
    pred: (n_samples, ) np.array
        Vectors of predictions (same order as labels).
    """
    
    #cv_folds = cross_validation.StratifiedKFold(labels, n_folds, shuffle=True)
    skf = model_selection.StratifiedKFold(n_folds, shuffle=True)
    cv_folds= skf.split(design_matrix, labels)
    
    pred = np.zeros(labels.shape)
    prob = np.zeros(labels.shape)
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

        # Predict probabilities on test data
        if type(classifier) not in [sklearn.cluster.k_means_.KMeans, sklearn.svm.classes.LinearSVC]:
            ytest_prob = classifier.predict_proba(Xtest)
            index_of_class_1 = 1-classifier.classes_[0]  # 0 if the first sample is positive, 1 otherwise
            prob[test_fold] = ytest_prob[:, index_of_class_1]
            
        ytest_pred = classifier.predict(Xtest)
        pred[test_fold] = ytest_pred
        
    return pred, prob


def learn(design_matrix, mlMethod, list_param, n_folds):
    
    labels = np.loadtxt('train_y.csv',  delimiter=',', skiprows=1, usecols=range(1, 2)).astype('int')

    dimensions = list(len(param) for param in list_param[::-1])

    nb_total_combination = reduce(operator.mul, dimensions, 1)
    
    # cas où la méthode de ML ne prend aucun hyperparamètre en argument
    if(nb_total_combination==0):
        clf = mlMethod()
        ypred, yprob = cross_validate(design_matrix, labels, clf, n_folds)
        return [[]], [ypred], [yprob], clf
    
    list_theta=[0]*nb_total_combination
    list_ypred=[0]*nb_total_combination
    list_yprob=[0]*nb_total_combination
    
    n_params = len(list_param)
    
    # liste des tailles progressives par ajout de dimension. Ex : Matrice 3x4x5 =>  slices_size=[1,3,12]
    slices_size= [1]+[0 for i in range(n_params-1)]
    for k in range(n_params-1):
        slices_size[k+1]=slices_size[k]*len(list_param[k])
    
    for i in range(nb_total_combination):

        p = i+1
        theta=[0]*n_params
        
        # récupération des valeurs des hyperparamètres
        for k in range(n_params-1,-1,-1):
            
            factor = slices_size[k]
            quot= p//factor
            rest= p%factor
             
            if rest==0:
                 theta[k]=list_param[k][quot-1]
                 p-=factor*(quot-1)
            else:
                theta[k]=list_param[k][quot]
                p= rest
        
        clf = mlMethod(*theta)
        
        ypred, yprob = cross_validate(design_matrix, labels, clf, n_folds)
        list_theta[i]=theta
        list_ypred[i]=ypred
        list_yprob[i]=yprob
    
    mat_theta_reshape_dim = dimensions + [n_params]
    mat_theta = np.reshape(list_theta, mat_theta_reshape_dim)
    
    mat_ypred_yprob_reshape_dim = dimensions+ [len(design_matrix)]
    mat_ypred = np.reshape(list_ypred, mat_ypred_yprob_reshape_dim)
    mat_yprob = np.reshape(list_yprob, mat_ypred_yprob_reshape_dim)
    
    # permutation des matrices pour retrouver l'ordre initial des hyperparamètres
    permut = [i for i in range(len(np.shape(mat_theta))-2,-1,-1)]+[len(np.shape(mat_theta))-1]
    
    mat_theta = np.transpose(mat_theta, permut)
    mat_ypred = np.transpose(mat_ypred, permut)
    mat_yprob = np.transpose(mat_yprob, permut)
    
    return mat_theta, mat_ypred, mat_yprob, clf
      
def predict(design_matrix, classifier, save=False, name_save = None):
    labels_pred = classifier.predict(design_matrix)
    
    if save:
        with open(name_save + ".csv", "w", newline='') as csv_file:
            fieldnames=['id','sleep_stage']
            writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(labels_pred)):
                writer.writerow({'id': str(i),'sleep_stage': str(labels_pred[i])})
                
    return labels_pred

def visualizeResults(mat_theta, mat_ypred, mat_yprob, variable_hyperparam_id, variable_hyperparam_name, list_fixed_hyperparam_values_id):
    
    labels = np.loadtxt('train_y.csv',  delimiter=',', skiprows=1, usecols=range(1, 2)).astype('int')
    
    mat_theta_shape =  np.shape(mat_theta)
    mat_ypred_yprob_shape = np.shape(mat_ypred)

    permut = [variable_hyperparam_id] + [i for i in range(len(mat_theta_shape)) if i != variable_hyperparam_id]
    
    mat_theta = np.transpose(mat_theta, permut)
    mat_ypred = np.transpose(mat_ypred, permut)
    mat_yprob = np.transpose(mat_yprob, permut)

    index=[slice(mat_ypred_yprob_shape[variable_hyperparam_id])]+list_fixed_hyperparam_values_id
    
    f1_scores = [sklearn.metrics.f1_score(labels, ypred, average='macro') for ypred in mat_ypred[tuple(index)]]
    aurocs = [sklearn.metrics.auc(*sklearn.metrics.roc_curve(labels, ypred, pos_label=1)[0:2]) for ypred in mat_ypred[tuple(index)]]

    #cas où il n'y a pas d'hyperparamètre
    if len(mat_theta[0])==0 or np.shape(mat_theta)==(1,1):
        print("F1-score :", *f1_scores)
        print("AUROC :", *aurocs)
        print("\n")
        printConfusionMatrix(labels,  mat_ypred)
        
    else:
        list_variable_hyperparam_values = mat_theta[tuple(index)][:,variable_hyperparam_id]
        
        plotScore(variable_hyperparam_name ,list_variable_hyperparam_values, "F1 score", f1_scores)
        plotScore(variable_hyperparam_name ,list_variable_hyperparam_values, "AUROCS", aurocs)
    
def printConfusionMatrix(labels,  mat_ypred):
    conf_mat = sklearn.metrics.confusion_matrix(labels,  mat_ypred[0], labels=None, sample_weight=None).T
    print("Matrice de confusion : \n")
    print("\t\tTrue 0  True 1  True 2  True 3  True 4")
    for p in range(5):
        row_to_print="Predicted " +str(p)
        for t in range(5):
            row_to_print+="\t"+  str(conf_mat[p,t])
        print(row_to_print)

def plotScore(variable_hyperparam_name ,list_variable_hyperparam_values, score_name, scores):
    plt.figure()
    plt.plot(list_variable_hyperparam_values, scores, color='red')
    plt.xlabel(variable_hyperparam_name, fontsize=16)
    plt.ylabel('Cross-validated : ' + score_name, fontsize=16)
    plt.title(variable_hyperparam_name, fontsize=16)