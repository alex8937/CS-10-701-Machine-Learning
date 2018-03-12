import numpy as np

def compute_kernel(XTrain,XTest,bw,kernel_type):
    """
    XTrain: training data, size of nTrain * f
    XTest: test data, size of nTest * f
    bw: bandwidth
    K: kernel, sized of nTrain * nTest.
    K_i,j denotes the kernel value for ith training sample and jth test sample
    for rbf, it returns logK
    """

    nTrain,f = XTrain.shape
    nTest = XTest.shape[0]

    L2 = np.zeros((nTrain,nTest))
    for i in range(nTrain):
        for j in range(nTest):
            diff = (XTrain[i] - XTest[j]) / bw
            L2[i][j] = diff.T.dot(diff)

    
    if kernel_type == 'rbf':
        K = np.exp(- 0.5 * L2)
       
    elif kernel_type == 'boxcar':
        K = (L2 < 1) * 1
        
    elif kernel_type == 'Epanechnikov':
        K = (1 - L2)
   
        
    elif kernel_type == 'Quartic':
        K = (1 - L2) ** 2     
       
    return K