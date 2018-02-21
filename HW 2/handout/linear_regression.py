import numpy as np

def compute_weights(XTrain,yTrain,init_w,nIter,lr,rate):
    """
    % XTrain: training data, size of nTrain*f
    % yTrain: training values, size of nTrain*1
    % init_w: initialized weights, size of f*1
    % nIter: number of iterations in gradient descent
    % lr: learning rate
    % lambda: coefficient for penality term
    % w: weights obtained after gradient descent, size of f*1
    """
    w = init_w
    for i in range(nIter):
        gradient = XTrain.T.dot(XTrain.dot(w) - yTrain.ravel()) + rate * w
        w = w - lr * gradient
    return w
    
def linearRegression(XTest, w):
    """
    % XTrain: training data, size of nTrain*f
    % yTrain: training values, size of nTrain*1
    % XTest: test data, size of nTest*f
    % init_w: initialized weights, size of f*1
    % nIter: number of iterations in gradient descent
    % y: predicted values, size of nTest*1
    """
    return XTest.dot(w)