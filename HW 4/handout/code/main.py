from scipy.io import loadmat
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import *
from regression import *

data = loadmat('../Data/data.mat')
X, y = data['XTrain'], data['yTrain']

test_size = 500
random.seed(101)
test_index = random.sample(range(0, X.shape[0]), test_size)
train_index = np.array([True] * X.shape[0])
train_index[test_index] = False

XTest, yTest, XTrain, yTrain = X[test_index], y[test_index], X[train_index], y[train_index]
scaler = preprocessing.StandardScaler().fit(XTrain)
XTrain_scaled = scaler.transform(XTrain) 
XTest_scaled = scaler.transform(XTest)

yTest_pred = regression(XTrain_scaled,yTrain,XTest_scaled)
print(accuracy_score(yTest, yTest_pred.flatten()))