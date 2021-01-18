function [Mu, C, EigenVectors, EigenValues] = compute_pca(X)
%COMPUTE_PCA Step-by-step implementation of Principal Component Analysis
%   In this function, the student should implement the Principal Component 
%   Algorithm
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o Mu              : (N x 1), Mean Vector of Dataset
%       o C               : (N x N), Covariance matrix of the dataset
%       o EigenVectors    : (N x N), Eigenvectors of Covariance Matrix.
%       o EigenValues     : (N x 1), Eigenvalues of Covariance Matrix

    %compute mean of each feature
    Mu = mean(X,2);
    
    %center the data
    X = X-Mu;
    
    %compute covariance matrix C
    C = (X*X.')/(size(X,2)-1);
    
    %retrieve eigenvectors and eigenvalues of C
    [V,D] = eig(C);
    EigenVectors = V;
    EigenValues = diag(D);
    
    %sort eigenvalues and eigenvectors
    [EigenValues,idx] = sort(EigenValues, 'descend');
    EigenVectors = EigenVectors(:,idx);
end

