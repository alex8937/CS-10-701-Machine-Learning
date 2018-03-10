function y = regression(XTrain,yTrain,XTest)
    % this is a wrapper function for kernel_regression
    % choose best kernel and bandwidth to get high score
    nTest = size(XTest,1);
    y = zeros(nTest,1);
    
    % comment this out if you want to see grades 
    % for other problems when you get a timeout
    % y = kernel_regression(XTrain,yTrain,XTest,1,'rbf');

end
