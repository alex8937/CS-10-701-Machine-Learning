from sklearn.metrics import *
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from compute_kernel import *
from KDE import *

def regression(XTrain,yTrain,XTest):
    """
    % this is a wrapper function for kernel_regression
    % choose best kernel and bandwidth to get high score
    """
    
    kf = KFold(n_splits= 10 , shuffle = True)
    accuracy_scores_matrix = []
    bws = np.logspace(-5, 5, num = 21)
    for regressor in [RBFRegression, BoxCarRegression, EpanechnikovRegression, QuarticRegression]:
        print(regressor.__name__)
        accuracy_scores = []
        for bw in bws: 
            print(bw)
            accuracy_scores.append([])
            for train, valid in kf.split(XTrain):
                yvalid = regressor(XTrain[train], yTrain[train], XTrain[valid], bw)
                acc = max(accuracy_score(yvalid, yTrain[valid]), 0)
                accuracy_scores[-1].append(acc)   
                print(regressor.__name__, acc)
        pd.DataFrame(accuracy_scores).plot(x = bws, logx = True, title = '{} CV'.format(regressor.__name__))    
        accuracy_scores_matrix.append(np.mean(accuracy_scores, 1))
        
    accuracy_scores_matrix = np.array(accuracy_scores_matrix)    
    best_bw = np.unravel_index(accuracy_scores_matrix.argmax(), accuracy_scores_matrix.shape)  
    regressor, bw = [RBFRegression, BoxCarRegression, EpanechnikovRegression, QuarticRegression][best_bw[0]], bws[best_bw[1]]  

    yTest_pred =  regressor(XTrain, yTrain, XTest, bw)  
    return yTest_pred
    
    
def RBFRegression(XTrain,yTrain,XTest, bw):
    K = compute_kernel(XTrain, XTest, bw, 'rbf')
    Ksum = KDE(XTrain, XTest, bw, 'rbf')
    y = (K.T.dot(yTrain) / Ksum)
    y[np.isnan(y)] = np.mean(yTrain)
    return np.round(y)

def BoxCarRegression(XTrain,yTrain,XTest, bw):
    K = compute_kernel(XTrain, XTest, bw, 'boxcar')
    Ksum = KDE(XTrain, XTest, bw, 'boxcar')
    y = (K.T.dot(yTrain) / Ksum)
    y[np.isnan(y)] = np.mean(yTrain)
    return np.round(y)

def EpanechnikovRegression(XTrain,yTrain,XTest, bw):
    K = compute_kernel(XTrain, XTest, bw, 'Epanechnikov')
    Ksum = KDE(XTrain, XTest, bw, 'Epanechnikov')
    y = (K.T.dot(yTrain) / Ksum)
    y[np.isnan(y)] = np.mean(yTrain)
    return np.round(y)

def QuarticRegression(XTrain,yTrain,XTest, bw):
    K = compute_kernel(XTrain, XTest, bw, 'Quartic')
    Ksum = KDE(XTrain, XTest, bw, 'Quartic')
    y = (K.T.dot(yTrain) / Ksum)
    y[np.isnan(y)] = np.mean(yTrain)
    return np.round(y)

    
    
