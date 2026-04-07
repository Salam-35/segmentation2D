%% the loaded data
clear all;
clc;

% Read subfolder data files, which corresponds to different classes
Data_file = 'GrayData';
Data_file_sub=dir(fullfile(Data_file,'\*'));
Data_file_sub_cell = struct2cell(Data_file_sub);
ClassNames= Data_file_sub_cell(1,3:end);
DataNew = 'RawData';
mkdir(DataNew)
Xsize = 256; % 
Ysize = 256; % 

% for i=1:length(ClassNames)
for i=1:2
    
    % read all images 
    ClassImages = dir(fullfile(Data_file,ClassNames{i},'\*.png'));
    N{i} = length(ClassImages);
    OutDir = fullfile(DataNew,ClassNames{i});
    mkdir(OutDir);
    
    image_index = zeros(N{i},1);
    for m = 1:N{i}
        image_index(m,1) = str2num(cell2mat(extractBetween(ClassImages(m).name,'(',')')));
    end
    [~,image_index] = sort(image_index);
    
    for m = 1:N{i}
        F = fullfile(ClassImages(image_index(m)).folder,ClassImages(image_index(m)).name);
        % read image
        I1 = imread(F);
        if ndims(I1)==3
            I1=rgb2gray(I1);
        end
        if i==1
            % equalize  image
            I2 = adapthisteq(I1,'NumTiles',[8,8],'clipLimit',0.01,'NBins',256,'Range','full','Distribution','rayleigh');
            % create image complement
            I3 = imcomplement(I1);
            % concatinate original, equalized and complement
            I = cat(3,I1,I2,I3);
        else
            I = I1;
        end
        % save image
%         imgNameGr = fullfile(OutDir,strcat(ClassNames{i},' (',num2str(m),').png'));
        imgNameGr = fullfile(OutDir,ClassImages(image_index(m)).name);
        I = imresize(I,[Ysize Xsize]);
        I = uint8(I);
        imwrite(I,imgNameGr);
    end
    
end