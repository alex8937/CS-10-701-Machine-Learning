from scipy.io import loadmat

from naive_bayes import *
from knn import *
from linear_regression import *

accuracy = lambda pred, actual: sum(pred.ravel() == actual.ravel()) / len(pred)

XTrain = loadmat('./data/Iris/XTrainIris.mat')['XTrain']
yTrain = loadmat('./data/Iris/yTrainIris.mat')['yTrain']
XTest = loadmat('./data/Iris/XTestIris.mat')['XTest']
yTest = loadmat('./data/Iris/yTestIris.mat')['yTest']

yTestPred = knn( XTrain, yTrain, XTest )
print('knn:', accuracy(yTestPred, yTest))

yTestPred = naiveBayesClassify( XTrain, yTrain, XTest )
print('NB:', accuracy(yTestPred, yTest))

XTrain = loadmat('./data/WBC/XTrainWBC.mat')['XTrain']
yTrain = loadmat('./data/WBC/yTrainWBC.mat')['yTrain']
XTest = loadmat('./data/WBC/XTestWBC.mat')['XTest']
yTest = loadmat('./data/WBC/yTestWBC.mat')['yTest']

yTestPred = knn( XTrain, yTrain, XTest )
print('knn:', accuracy(yTestPred, yTest))

yTestPred = naiveBayesClassify( XTrain, yTrain, XTest )
print('NB:', accuracy(yTestPred, yTest))

XTrain = loadmat('./data/Challenge/XTrain.mat')['XTrain']
yTrain = loadmat('./data/Challenge/yTrain.mat')['yTrain']

