function [ Sigma ] = compute_covariance( X, X_bar, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o X_bar : (N x 1), an Nx1 matrix corresponding to mean of data X
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%

[N,M] = size(X);
X = X - repmat(X_bar,1,M);

switch type
    
    case "full"
        Sigma = X*X.'/(M-1);

        
    case "diag"
        Sigma = diag( diag(X*X.'/(M-1)) );
        
    case "iso"
        squared_norms = vecnorm(X).^2;
        sigma_iso = sum(squared_norms)/(N*M);
        Sigma = eye(N)*sigma_iso;

end

end

