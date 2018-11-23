from sklearn import cross_validation

folds = cross_validation.StratifiedKFold(y, 10, shuffle=True)

from sklearn import preprocessing
def cross_validate(design_matrix, labels, classifier, cv_folds):
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
    pred = np.zeros(labels.shape)
    prob = np.zeros(labels.shape)
    for train_folds, test_fold in cv_folds:
        
        # Restrict data to train/test folds
        Xtrain = design_matrix[train_folds, :]
        ytrain = labels[train_folds]
        Xtest = design_matrix[test_fold, :]

        # Scale data
        scaler = preprocessing.StandardScaler() # create scaler
        Xtrain = scaler.fit_transform(Xtrain) # fit the scaler to the training data and transform training data
        Xtest = scaler.transform(Xtest) # transform test data
        
        # Fit classifier
        classifier.fit(Xtrain, ytrain)

        # Predict probabilities (of belonging to +1 class) on test data
        ytest_prob = classifier.predict_proba(Xtest)
        print(ytest_prob)
        ytest_pred = classifier.predict(Xtest)
        print(ytest_pred)
        index_of_class_1 = 1-classifier.classes_[0]  # 0 if the first sample is positive, 1 otherwise
        
        prob[test_fold] = ytest_prob[:, index_of_class_1]
        pred[test_fold] = ytest_pred
    return pred, prob