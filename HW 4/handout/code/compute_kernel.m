function K = compute_kernel(XTrain,XTest,bw,kernel_type)
    % XTrain: training data, size of nTrain * f
    % XTest: test data, size of nTest * f
    % bw: bandwidth
    % K: kernel, sized of nTrain * nTest. 
    % K_i,j denotes the kernel value for ith training sample 
    % and jth test sample

    [nTrain,f] = size(XTrain);
    nTest = size(XTest,1);
    K = zeros(nTrain,nTest);
    if strcmp(kernel_type,'rbf')
        %% begin

        %% end
    elseif strcmp(kernel_type,'boxcar')
        %% begin

        %% end
    elseif strcmp(kernel_tye,'my_kernel')
        
    end
end
