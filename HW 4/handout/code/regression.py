from sklearn.metrics import *
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from compute_kernel import *

def regression(XTrain,yTrain,XTest):
    """
    % this is a wrapper function for kernel_regression
    % choose best kernel and bandwidth to get high score
    """
    
    kf = KFold(n_splits= 10 , shuffle = True)
    accuracy_scores_matrix = []
    bws = np.logspace(-5, 5, num = 10)
    for regressor in [RBFRegression, BoxCarRegression, EpanechnikovRegression, QuarticRegression]:
        print(regressor.__name__)
        accuracy_scores = []
        for bw in bws: 
            print(bw)
            accuracy_scores.append([])
            for train, valid in kf.split(XTrain):
                yvalid = regressor(XTrain[train], yTrain[train], XTrain[valid], bw)
                acc = accuracy_score(yvalid, yTrain[valid].flatten())
                accuracy_scores[-1].append(acc)     
        pd.DataFrame(accuracy_scores).plot(x = bws, logx = True, title = '{} CV'.format(regressor.__name__))    
        accuracy_scores_matrix.append(np.mean(accuracy_scores, 1))
        
    accuracy_scores_matrix = np.array(accuracy_scores_matrix)    
    best_bw = np.unravel_index(accuracy_scores_matrix.argmax(), accuracy_scores_matrix.shape)  
    regressor, bw = [RBFRegression, BoxCarRegression, EpanechnikovRegression, QuarticRegression][best_bw[0]], bws[best_bw[1]]  

    yTest_pred =  regressor(XTrain, yTrain, XTest, bw)  
    return yTest_pred
    
    
def RBFRegression(XTrain,yTrain,XTest, bw):
    logK = compute_kernel(XTrain, XTest, bw, 'rbf')
    logy = np.log(yTrain)
    y = np.round(np.sum(np.exp(logK + logy), 0))
    return y

def BoxCarRegression(XTrain,yTrain,XTest, bw):
    K = compute_kernel(XTrain, XTest, bw, 'boxcar')
    return np.round(K.T.dot(yTrain))

def EpanechnikovRegression(XTrain,yTrain,XTest, bw):
    K = compute_kernel(XTrain, XTest, bw, 'Epanechnikov')
    return np.round(K.T.dot(yTrain))

def QuarticRegression(XTrain,yTrain,XTest, bw):
    K = compute_kernel(XTrain, XTest, bw, 'Quartic')
    return np.round(K.T.dot(yTrain))

    
    
