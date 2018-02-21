function [ y ] = naiveBayesClassify( XTrain, yTrain, XTest )
    % XTrain: training data, size of nTrain*f
    % yTrain: training labels, size of nTrain*1
    % XTest: test data, size of nTest*f
    
    [nTrain,f] = size(XTrain);
    nTest = size(XTest,1);
    y = zeros(nTest,1);
    class = unique(yTrain);
    nClass = size(class,1);
    
    %% begin

    %% end
end



