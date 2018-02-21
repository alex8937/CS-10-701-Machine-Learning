function yTest = knn(XTrain,yTrain,XTest)
    % XTrain: training data, nTrain*f matrix
    % yTrain: training labels, nTrain*1 matrix
    % XTest: test data, nTest*f matrix
    
    % modify 'k' and 'd' to get the best accuracy
    k = 3;
    d = 'euclidean';
    yTest = knnClassify(XTrain,yTrain,XTest,k,d);
end

