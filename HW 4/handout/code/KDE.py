import numpy as np

def KDE(XTrain,XTest,bw,kernel_type):
    """
      XTrain: training data, size of nTrain * f
      XTest: test data, size of nTest * f
      bw: bandwidth
      p: estimated density for test data, size nTest * 1
    """
    
    nTrain,f = XTrain.shape
    nTest = XTest.shape[0]
    p = np.zeros((nTest,1))

