%% the loaded data
clear all;
clc;
%%%%%%%%%%%%% make changes to this block only %%%%%%%%%%%%%%%%%%%
% kindly make sure that:
%   1) Dataset images are in png fromat
%   2) Dataset is well shuffled
%   3) Dataset folders are in the following format:
%       Data/
%           /images
%               images_name (1).png
%               images_name (2).png
%                      .
%                      .
%               images_name (m).png
%           /masks
%               images_name (1).png
%               images_name (2).png
%                      .
%                      .
%               images_name (m).png
%    4) Each image and it corresponding mask have the same name
num_folds = 5;                   % number of folds
val_ratio =0.2;                  % percentage of validation out of the full train set, please set to zero if no val
gray_scale = false;               % true: convert all images to gray scale, false: keep the original format
Xsize = 224;                     % rescaled image width
Ysize = Xsize;                   % rescaled image height
Data_folder = 'RawData';         % Raw data folder, with a sub folder for each class of images
AugmentedData_folder = 'Data';   % Data folder to save Train,Val and Test Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read subfolder data files, which corresponds to different classes
ClassNames=dir(fullfile(Data_folder,'\*'));
ClassNames = struct2cell(ClassNames);
ClassNames= ClassNames(1,3:end);

% create augmented data directories
if ~exist(AugmentedData_folder, 'dir')
    mkdir(AugmentedData_folder)
end
TrainFile = fullfile(AugmentedData_folder,'Train');
if ~exist(TrainFile, 'dir')
    mkdir(TrainFile);
end
TestFile = fullfile(AugmentedData_folder,'Test');
if ~exist(TestFile, 'dir')
    mkdir(TestFile);
end
if val_ratio~=0
    ValFile = fullfile(AugmentedData_folder,'Val');
    if ~exist(ValFile, 'dir')
        mkdir(ValFile);
    end
end

% iterate through classes
for i=1:length(ClassNames)
    % i=1 >> images
    % i=2 >> masks
    if i==2
        gray_scale = true;
    end
    % read all images names for a specific class
    ClassImages = dir(fullfile(Data_folder,ClassNames{i},'\*.png'));
    N{i} = length(ClassImages);
    % sort images as in folder
    [m,~]=size(ClassImages);
    image_index = zeros(m,1);
    for k = 1:m
        image_index(k,1) = str2double(cell2mat(extractBetween(ClassImages(k).name,'(',')')));
    end
    [~,image_index] = sort(image_index);
    ClassImages = ClassImages(image_index);
    % generate train and test indexes
%     [TestFolds_idx,TrainFolds_idx] = CV_Index(num_folds,N{i});
    if val_ratio ~= 0
        [TestFolds_idx,TrainFolds_idx,ValFolds_idx] = CV_IndexVal(num_folds,N{i},val_ratio);
    else
        [TestFolds_idx,TrainFolds_idx] = CV_Index(num_folds,N{i});
    end
    % iterate through folds
    for k=1:num_folds
        
        %%% Test
        % make test folder for a specific fold
        FoldFile = fullfile(TestFile,sprintf('fold_%d',k),ClassNames{i});
        mkdir(FoldFile);
        % save test data
        SaveFold(ClassImages, TestFolds_idx{k}, FoldFile, Ysize, Xsize, gray_scale);
        
        %%% Validation
        if val_ratio ~= 0
            % split train data into train and validation
%             num_val_images = round(val_ratio*length(TrainFolds_idx{k}));
%             ValFolds_idx{k} = TrainFolds_idx{k}(1:num_val_images);
%             TrainFolds_idx{k} = TrainFolds_idx{k}(num_val_images+1:end);
            % make validation folder for a specific fold
            FoldFile = fullfile(ValFile,sprintf('fold_%d',k),ClassNames{i});
            mkdir(FoldFile);
            % save validation data
            SaveFold(ClassImages, ValFolds_idx{k}, FoldFile, Ysize, Xsize, gray_scale);
        end
        
        %%% Train
        % make train folder for a specific fold
        FoldFile = fullfile(TrainFile,sprintf('fold_%d',k),ClassNames{i});
        mkdir(FoldFile);
        % save train data
        TrainImages = ClassImages(TrainFolds_idx{k});
        % images counter
        x = 1;
        % concatiante dimension
        if gray_scale
            concatinate_dim = 3;
        else
            concatinate_dim = 4;
        end
        % iterate through training images
        for m=1:length(TrainImages)
            % read image
            F_load = fullfile(TrainImages(m).folder,TrainImages(m).name);
            I = imread(F_load);
            I = imresize(I,[Ysize Xsize]);
            % if rgb, then convert to gray
            if gray_scale
                if ndims(I)==3
                    I = rgb2gray(I);
                end
            end
            % Augment Data
            switch ClassNames{i}
                case 'images'
%                     Aug1 = uint8(imrotate(I,5,'nearest','loose'));
%                     Aug1 = imresize(Aug1,[Ysize Xsize]);
%                     Aug2 = uint8(imrotate(I,-5,'nearest','loose'));
%                     Aug2 = imresize(Aug2,[Ysize Xsize]);
%                     Aug3 = uint8(imtranslate(I,[-0.10*Xsize -0.10*Ysize]));
%                     AugData =cat(concatinate_dim,Aug1,Aug2,Aug3);
                                        AugData = [];
                case 'masks'
%                     Aug1 = uint8(imrotate(I,5,'nearest','loose'));
%                     Aug1 = imresize(Aug1,[Ysize Xsize]);
%                     Aug2 = uint8(imrotate(I,-5,'nearest','loose'));
%                     Aug2 = imresize(Aug2,[Ysize Xsize]);
%                     Aug3 = uint8(imtranslate(I,[-0.10*Xsize -0.10*Ysize]));
%                     AugData =cat(concatinate_dim,Aug1,Aug2,Aug3);
                                        AugData = [];
            end
            % save train data
            temp_name = extractBefore(ClassImages(m).name,'(');
            temp_num = extractBetween(ClassImages(m).name,'(',')');
            F_save = fullfile(FoldFile, strcat(temp_name,temp_num{:},' (',num2str(x),').png') );
            imwrite(I,F_save);
            x = x+1;
            if ~isempty(AugData)
                for j=1:size(AugData,concatinate_dim)
                    F_save = fullfile(FoldFile, strcat(temp_name,temp_num{:},'_aug (',num2str(x),').png'));
                    if gray_scale
                        imwrite(AugData(:,:,j),F_save);
                    else
                        imwrite(AugData(:,:,:,j),F_save);
                    end
                    x = x+1;
                end
            end
        end
        
        
    end
    
end








