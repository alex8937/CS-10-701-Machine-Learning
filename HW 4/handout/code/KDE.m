function [p] = KDE(XTrain,XTest,bw,kernel_type)
    % XTrain: training data, size of nTrain * f
    % XTest: test data, size of nTest * f
    % bw: bandwidth
    % p: estimated density for test data, size nTest * 1
    
    [nTrain,f] = size(XTrain);
    nTest = size(XTest,1);
    p = zeros(nTest,1);
    %% begin 

    %% end
end
