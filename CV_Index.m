function [TestFolds_idx,TrainFolds_idx] = CV_Index(num_folds,N)
%     TestIndex = crossvalind('Kfold',N,num_folds);
    fold_images = round(N/num_folds);
    TestIndex =[];
    for i=1:num_folds-1
        TestIndex = [TestIndex; i*ones(fold_images,1)];
    end
    i=i+1; 
    TestIndex = [TestIndex; i*ones(N-length(TestIndex),1)];
    TestFolds_idx={};
    TrainFolds_idx={};
    data_idx=1:N;
    for j=1:num_folds
        test_idx = find(TestIndex==j);
        TestFolds_idx{j}= test_idx';
        TrainFolds_idx{j} = data_idx( ~ismember( data_idx, test_idx ) );
    end      
end

