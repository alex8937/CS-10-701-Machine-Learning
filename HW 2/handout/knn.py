import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import KFold
from collections import Counter

def knn(XTrain,yTrain,XTest):
    kf = KFold(n_splits= 10 , shuffle = True)
    best_score = 0
    for dist in [L1_norm, L2_norm, KL_divergence]:
        for k in range(1, 20):
            scores = []
            for train, test in kf.split(XTrain):
                score = sum(yTrain[test].ravel() == knnClassify(XTrain[train],yTrain[train],XTrain[test],k,dist)) / len(test)
                scores.append(score)
            average_score = np.mean(scores)
            if best_score < average_score:
                best_score, best_k, best_dist = average_score, k, dist
    print(best_score, best_k, best_dist)            
    return knnClassify(XTrain,yTrain,XTest,best_k,best_dist)
                

def compute_distance(XTrain, XTest, distance_fun):
    """
    % d: if d is 'euclidean', return the distance matrix
    %       you can implement your own distance meansurements (one or more)
    %       for the score board problem. Feel free to rename your distance
    %       metric.
    % D: distance matrix, the size should be nTrain*nTest, where D(i,j) is
    %    the distance of XTrain(i,:) and XTest(j,:)
    """
    nTrain = XTrain.shape[0]
    nTest = XTest.shape[0]
    D = np.zeros((nTest, nTrain))
    for i in range(nTrain):
        for j in range(nTest):
            D[j, i] = distance_fun(XTest[j], XTrain[i])
    return D           

def L1_norm(a, b):
    return np.linalg.norm((a - b), ord=1)

def L2_norm(a, b):
    return np.linalg.norm((a - b), ord=2)

def KL_divergence(a, b):
    return entropy(a, b)

def knnClassify(XTrain,yTrain,XTest,k,distance_fun):
    """
    % XTrain: training data
    % yTrain: training labels
    % XTest: test data
    % k: number of neighbors
    % yTest: predicted test labels
    % mode: 'euclidean' for euclidean distance
    %       you can implement other distance for leader board
    """
    D = compute_distance(XTrain, XTest, distance_fun)
    top_k_fun = lambda x : np.argpartition(x, k)[:k]
    top_k_label = yTrain.ravel()[np.apply_along_axis(top_k_fun, 1, D)]
    major_vote = lambda x : max(x, key=Counter(x).get)
    
    return np.apply_along_axis(major_vote, 1, top_k_label)

