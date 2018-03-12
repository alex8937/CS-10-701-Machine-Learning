import numpy as np
from compute_kernel import *

def KDE(XTrain,XTest,bw,kernel_type):
    """
      XTrain: training data, size of nTrain * f
      XTest: test data, size of nTest * f
      bw: bandwidth
      p: estimated density for test data, size nTest * 1
    """
    K = compute_kernel(XTrain,XTest,bw,kernel_type)
    Ksum = np.sum(K, 0)
        
    return Ksum.reshape(-1, 1)
     
   

