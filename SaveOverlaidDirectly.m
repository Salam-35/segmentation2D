clear
clc
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_points = 60;      %Number of points of the mask
mask_path = 'masks\';
image_path = 'images\';
newmask_path = 'overlaid ground-truth';
image_size = 256;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create the folders to save new images:
if ~exist(newmask_path, 'dir')
    mkdir(newmask_path)
end
data_mask = dir(mask_path);
data_mask = data_mask(3:end,:);
data_image = dir(image_path);
data_image = data_image(3:end,:);

for i=1:length(data_mask)
    
    M = imread(fullfile(mask_path,data_mask(i).name));  %Reading Mask
    M = imresize(M, [image_size,image_size]);
    M(M>0) = 255;
    I = imread(fullfile(image_path,data_image(i).name));
    I = imresize(I, [image_size,image_size]);
    if ndims(I)==3
        I = rgb2gray(I);
    end
    % find mask boundries
    BW = im2bw(M);
    BW_filled = imfill(BW,'holes');
    boundaries = bwboundaries(BW_filled);
    
    if ~isempty(boundaries)
        A = [];
        for k=1:length(boundaries)
            b = boundaries{k};
            A = [A; length(boundaries{k})];
        end
        A1 = max(A);
        u=0;
        for s=1:length(A)
            if(A(s)==A1)
                u=u+1;
            end
        end
        if(u==1) && length(A)==1        %If only one object in A exists, make A2 and A1 same
            A2=A1;
            x1 = find(A==A1);
            x2 =x1;
        elseif(u==2)
            A2=A1;
            temp_index = find(A==A1);
            x1 = temp_index(1);
            x2 = temp_index(2);
        else
            A2=A1;
            A2 = max(A(A<max(A)));
            x1 = find(A==A1); x2 = find(A==A2);
        end
        A1 = boundaries{x1};
        A2 = boundaries{x2};
        if length(A1)< 3
            A1=[1 1; 5 5; 10 10; 15 15; 20 20];
        end
        if length(A2)< 3
            A2=[25 25 ; 30 30; 35 35; 40 40; 45 45];
        end
    else
        A1=[1 1; 5 5; 10 10; 15 15; 20 20];
        A2=[25 25 ; 30 30; 35 35; 40 40; 45 45];
    end
    %
    if length(A1)>N_points  %%Check if code could not find boundaries
        step = length(A1)/N_points;
        step = ceil(step);
        %
        index = 1;
        j=1;
        while index(end)<length(A1)
            index = [index; j*step];
            j = j + 1;
        end
        if index(end)> length(A1)
            index(end)= length(A1);
        end
        newlung1 = A1(index,:);
        temp = newlung1;
        newlung1(:,1) = temp(:,2);
        newlung1(:,2) = temp(:,1);
        %
    else
        newlung1=A1;
    end
    %
    if length(A2)>N_points  %%Check if code could not find boundaries
        step2 = length(A2)/N_points;
        step2=ceil(step2);
        %
        index2 = 1;
        j=1;
        while index2(end)<length(A2)
            index2 = [index2; step2*j];
            j = j + 1;
        end
        if index2(end)> length(A2)
            index2(end)= length(A2);
        end
        newlung2 = A2(index2,:);
        temp2 = newlung2;
        newlung2(:,1) = temp2(:,2);
        newlung2(:,2) = temp2(:,1);
    else
        newlung2=A2;
    end
    
    
    % display image
    I_mask = cat(3, I, I, I);
    for jj=1:size(A1,1)
        I_mask(A1(jj,1), A1(jj,2), :)=[0 255 0];
    end
    for jj=1:size(A2,1)
        I_mask(A2(jj,1), A2(jj,2), :)=[0 255 0];
    end
    % plot
    imwrite(I_mask,strcat(newmask_path, '\', data_mask(i).name));
    
end