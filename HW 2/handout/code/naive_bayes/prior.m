function p = prior(yTrain)
    % yTrain: training labels, size of n*1
    % p: prior, size of k*1, where k is number of classes
    class = unique(yTrain);
    nClass = size(class,1);
    p = zeros(nClass,1);
    nTrain = size(yTrain,1);
    %% begin

    %% end
    
end
