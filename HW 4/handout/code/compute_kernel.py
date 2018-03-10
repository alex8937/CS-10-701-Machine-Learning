import numpy as np

def compute_kernel(XTrain,XTest,bw,kernel_type):
    """
    XTrain: training data, size of nTrain * f
    XTest: test data, size of nTest * f
    bw: bandwidth
    K: kernel, sized of nTrain * nTest.
    K_i,j denotes the kernel value for ith training sample and jth test sample
    """

    nTrain,f = XTrain.shape
    nTest = XTest.shape[0]

    L2 = np.zeros((nTrain,nTest))
    for i in range(nTrain):
        for j in range(nTest):
            diff = (XTrain[i] - XTest[j]) / bw
            L2[i][j] = diff.T.dot(diff)

    
    if kernel_type == 'rbf':
        logh = - 0.5 * ( f * np.log(2 * np.pi) + L2)
        logK = -f * np.log(bw) + logh
        logKmax = np.max(logK, axis = 0).reshape(-1, 1)
        logKdiff = (logK.T - logKmax).T
        K = (logKdiff.T - np.log(np.sum(np.exp(logKdiff), axis = 0)).reshape(-1, 1)).T
       
    elif kernel_type == 'boxcar':
        K = L2 * (L2 < 1)
        Ksum = np.sum(K, axis = 0)
        if np.any(Ksum == 0):
            print("Boxcar Warning: too small bw for boxcar")
        K = K / (Ksum + 1e-100)
        
    elif kernel_type == 'Epanechnikov':
        K = 0.75 * (1 - L2)
        Ksum = np.sum(K, axis = 0)
        if np.any(Ksum == 0):
            print("Warning: too small bw for Epanechnikov")
        K = K / (Ksum + 1e-100)     
        
    elif kernel_type == 'Quartic':
        K = (15 / 16) * (1 - L2) ** 2
        Ksum = np.sum(K, axis = 0)
        if np.any(Ksum == 0):
            print("Warning: too small bw for Quartic")
        K = K / (Ksum + 1e-100)           
       
    return K