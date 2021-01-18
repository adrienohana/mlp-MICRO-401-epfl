function [cimg, ApList, muList] = compress_image(img, p)
%COMPRESS_IMAGE Compress the image by applying the PCA over each channels 
% independently
%
%   input -----------------------------------------------------------------
%   
%       o img : (width x height x 3), an image of size width x height over RGB channels
%       o p : The number of components to keep during projection 
%
%   output ----------------------------------------------------------------
%
%       o cimg : (p x height x 3) The projection of the image on the eigenvectors
%       o ApList : (p x width x 3) The projection matrices for each channels
%       o muList : (width x 3) The mean vector for each channels

%compute pca for each color and project pca to compress images
for i=1:size(img,3)
    X = img(:,:,i);
    [Mu, C, EigenVectors, EigenValues] = compute_pca(X);
    [Y, Ap] = project_pca(X, Mu, EigenVectors, p);
    
    cimg(:,:,i) = Y;
    ApList(:,:,i) = Ap;
    muList(:,i)= Mu;
end
end

