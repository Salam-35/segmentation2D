function SaveFold(ClassImages, Fold_idx, FoldFile, Ysize, Xsize, gray_scale) 
    ClassImages = ClassImages(Fold_idx);
    for i=1:size(ClassImages,1)
        % read image
        F_load = fullfile(ClassImages(i).folder,ClassImages(i).name);
        I = imread(F_load);
        I = imresize(I,[Ysize Xsize]); 
        % convert to gray  
        if gray_scale
            if ndims(I)==3
                I = rgb2gray(I);
            end
        end 
        % save image
        F_save = fullfile(FoldFile,ClassImages(i).name);
        imwrite(I,F_save);
    end
end

