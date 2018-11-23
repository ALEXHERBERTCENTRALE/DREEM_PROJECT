import numpy as np
import sklearn
from functools import reduce
import operator
import matplotlib.pyplot as plt

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
    
    cv_folds = sklearn.cross_validation.StratifiedKFold(labels, n_folds, shuffle=True)
    
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

        # Predict probabilities (of belonging to +1 class) on test data
        ytest_prob = classifier.predict_proba(Xtest)
        ytest_pred = classifier.predict(Xtest)

        index_of_class_1 = 1-classifier.classes_[0]  # 0 if the first sample is positive, 1 otherwise
        
        prob[test_fold] = ytest_prob[:, index_of_class_1]
        pred[test_fold] = ytest_pred
    return pred, prob


def learnAndPredict(design_matrix, mlMethod, list_param, n_folds):
    dimensions = list(len(param) for param in list_param)
    nb_total_combination = reduce(operator.mul, dimensions, 1)
    
    list_theta=[0]*nb_total_combination
    list_ypred=[0]*nb_total_combination
    list_yprob=[0]*nb_total_combination
    
    labels = np.loadtxt('train_y.csv',  delimiter=',', skiprows=1, usecols=range(1, 2)).astype('int')
    
    n_params = len(list_param)
    for i in range(nb_total_combination):
        p = i
        theta=[0]*n_params
        for k in range(n_params):
            
            p = p%len(list_param[k])
            theta[k]=list_param[k][p]
        clf = mlMethod(*theta)
        
        ypred, yprob = cross_validate(design_matrix, labels, clf, n_folds)
        list_theta[i]=theta
        list_ypred[i]=ypred
        list_yprob[i]=yprob
    mat_theta = np.reshape(list_theta, dimensions)
    mat_ypred = np.reshape(list_ypred, dimensions)
    mat_yprob = np.reshape(list_yprob, dimensions)
    return mat_theta, mat_ypred, mat_yprob
        

def visualizeResults(mat_theta, mat_ypred, mat_yprob, variable_hyperparam_id, variable_hyperparam_name, list_fixed_hyperparam_values_id):
    
    labels = np.loadtxt('train_y.csv',  delimiter=',', skiprows=1, usecols=range(1, 2)).astype('int')
    
    dimensions = np.shape(mat_ypred)
    permut = [variable_hyperparam_id] + [i for i in range(len(dimensions)) if i != variable_hyperparam_id]
    mat_theta = np.transpose(mat_theta, permut)
    mat_ypred = np.transpose(mat_ypred, permut)
    mat_yprob = np.transpose(mat_yprob, permut)

    index=[slice(dimensions[variable_hyperparam_id])]+list_fixed_hyperparam_values_id
    
    f1_scores = [sklearn.metrics.f1_score(labels, ypred, average='macro') for ypred in mat_ypred[tuple(index)]]
    aurocs = [sklearn.metrics.auc(sklearn.metrics.roc_curve(labels, ypred, pos_label=1)[0:2]) for ypred in mat_ypred[tuple(index)]]
    
    list_variable_hyperparam_values = mat_theta[tuple(index)]
    
    plotScore(variable_hyperparam_name ,list_variable_hyperparam_values, "F1 score", f1_scores)
    plotScore(variable_hyperparam_name ,list_variable_hyperparam_values, "AUROCS", aurocs)
    
    
def plotScore(variable_hyperparam_name ,list_variable_hyperparam_values, score_name, scores):
    plt.figure()
    plt.plot(list_variable_hyperparam_values, scores, color='red')
    plt.xlabel(variable_hyperparam_name, fontsize=16)
    plt.ylabel('Cross-validated : ' + score_name, fontsize=16)
    plt.title(variable_hyperparam_name, fontsize=16)
    

        
    