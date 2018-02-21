function D = compute_distance(XTrain,XTest,d)
    % d: if d is 'euclidean', return the distance matrix
    %       you can implement your own distance meansurements (one or more)
    %       for the score board problem. Feel free to rename your distance
    %       metric.
    % D: distance matrix, the size should be nTrain*nTest, where D(i,j) is
    %    the distance of XTrain(i,:) and XTest(j,:)
    [nTrain,f] = size(XTrain);
    nTest = size(XTest,1);
    D = zeros(nTrain,nTest);
    if strcmp(d,'euclidean')
    %% Begin
       
    %% End
    elseif strcmp(d,'other_dist')
    %% Begin
        
    %% End
    end

end