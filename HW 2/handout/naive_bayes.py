import numpy as np
import scipy.stats

def naiveBayesClassify( XTrain, yTrain, XTest ):
    """
    % XTrain: training data, size of nTrain*f
    % yTrain: training labels, size of nTrain*1
    % XTest: test data, size of nTest*f
    """
    nTrain, nFeature = XTrain.shape;
    nTest = XTest.shape[0]
    y = np.zeros((nTest, 1))
    label = np.unique(yTrain)
    nLabel = len(label)
    
    p = prior(yTrain)
    mu, sigma = likelihood( XTrain, yTrain )
    
    for i in range(nTest):
        post = np.copy(p)
        for k in range(nLabel):
            for f in range(nFeature):
                mu_fk, sigma_fk = mu[f, k], sigma[f, k]
                post[k] += scipy.stats.norm(mu_fk, sigma_fk).logpdf(XTest[i, f])
        y[i, 0] = label[np.argmax(post)]
    return y

def likelihood( XTrain, yTrain ):
    """
    % XTrain: Training data, size of n*f.
    % yTrain: Training labels, size of n*1.
    % mu: Mean of P(X_i|y_k), where i denotes ith feature 
    %     and k denotes kth class
    % sigma: Standard deviation of P(X_i|y_k)
    """
    
    label = np.unique(yTrain)
    nLabel = len(label)
    nTrain, nFeature = XTrain.shape
    mu = np.zeros((nFeature, nLabel))
    sigma = np.zeros((nFeature, nLabel))
    
    for l in range(nLabel):
        for f in range(nFeature):
            X = XTrain[:, f][yTrain[:, 0] == label[l]]
            mu[f, l] = np.mean(X)
            sigma[f, l] = np.std(X)

    return mu, sigma

def prior(yTrain):
    """ 
        yTrain: training labels, size of n*1
        p: prior, size of k*1, where k is number of classes
    """
    label, count = np.unique(yTrain, return_counts=True)
    p = np.log(count / len(yTrain))
    return p

