function [cr, compressedSize] = compression_rate(img,cimg,ApList,muList)
%COMPRESSION_RATE Calculate the compression rate based on the original
%image and all the necessary components to reconstruct the compressed image
%
%   input -----------------------------------------------------------------
%       o img : The original image   
%       o cimg : The compressed image
%       o ApList : List of projection matrices for each independent
%       channels
%       o muList : List of mean vectors for each independent channels
%
%   output ----------------------------------------------------------------
%
%       o cr : The compression rate
%       o compressedSize : The size of the compressed elements

%64 bits per element
s_Y = 64*numel(cimg);
s_Ap = 64*numel(ApList);
s_Mu = 64*numel(muList);
s_img = 64*numel(img);

%apply compression rate formula
cr = 1-(s_Y+s_Ap+s_Mu)/s_img;

%size after compression
compressedSize = (s_Y+s_Ap+s_Mu);
% convert the size to megabits
compressedSize = compressedSize/1048576; 
end

