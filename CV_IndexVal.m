function [TestFolds_idx,TrainFolds_idx,ValFolds_idx] = CV_IndexVal(num_folds,N,val_ratio)
%     TestIndex = crossvalind('Kfold',N,num_folds);
    fold_images = round(N/num_folds);
    TestIndex =[];
    for i=1:num_folds-1
        TestIndex = [TestIndex; i*ones(fold_images,1)];
    end
    i=i+1; 
    TestIndex = [TestIndex; i*ones(N-length(TestIndex),1)];
    TestFolds_idx = cell(1,num_folds);
    TrainFolds_idx = cell(1,num_folds);
    data_idx=1:N;
    for j=1:num_folds
        test_idx = find(TestIndex==j);
        TestFolds_idx{j}= test_idx';
        TrainFolds_idx{j} = data_idx( ~ismember( data_idx, test_idx ) );
    end    
    
    %%% Validation
    ValFolds_idx = cell(1,num_folds);
    % iterate through folds
    for k=1:num_folds
        % split train data into train and validation
        num_val_images = round(val_ratio*length(TrainFolds_idx{k}));
        last_test = TestFolds_idx{k}(end);
        last_train = TrainFolds_idx{k}(end);
        s = last_test +1;
        if s > last_train
            s = 1;
        end
        last_to_cover =  s + num_val_images -1;
        if last_to_cover <= last_train
            ValFolds_idx{k} = s:last_to_cover;
        else
            range_1_to_cover = last_train - s + 1;
            ValFolds_idx{k} = s:last_train;
            range_2_to_cover = num_val_images - range_1_to_cover;
            ValFolds_idx{k} = 1:range_2_to_cover;
        end
        [~,val_loc] = ismember(ValFolds_idx{k},TrainFolds_idx{k});
        TrainFolds_idx{k}(val_loc) = [];
%         TestFolds_idx{k} = TestFolds_idx{k} -1;
%         ValFolds_idx{k} = ValFolds_idx{k} -1;
%         TrainFolds_idx{k} = TrainFolds_idx{k}-1;
    end
end

