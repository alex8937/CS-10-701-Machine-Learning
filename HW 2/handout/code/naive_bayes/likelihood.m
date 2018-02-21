function [ mu, sigma ] = likelihood( XTrain, yTrain )

    % XTrain: Training data, size of n*f.
    % yTrain: Training labels, size of n*1.
    % mu: Mean of P(X_i|y_k), where i denotes ith feature 
    %     and k denotes kth class
    % sigma: Standard deviation of P(X_i|y_k)

    class = unique(yTrain);
    nClass = size(class,1);
    [nTrain,f] = size(XTrain);
    mu = zeros(f,nClass);
    sigma = zeros(f,nClass);

    %% begin

    %% end

end
