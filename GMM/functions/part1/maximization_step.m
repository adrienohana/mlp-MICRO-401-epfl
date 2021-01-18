function [Priors,Mu,Sigma] = maximization_step(X, Pk_x, params)
%MAXIMISATION_STEP Compute the maximization step of the EM algorithm
%   input------------------------------------------------------------------
%       o X         : (N x M), a data set with M samples each being of 
%       o Pk_x      : (K, M) a KxM matrix containing the posterior probabilty
%                     that a k Gaussian is responsible for generating a point
%                     m in the dataset, output of the expectation step
%       o params    : The hyperparameters structure that contains k, the number of Gaussians
%                     and cov_type the coviariance type
%   output ----------------------------------------------------------------
%       o Priors    : (1 x K), the set of updated priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu        : (N x K), an NxK matrix corresponding to the updated centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
%                   updated Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%



[N,M] = size(X);
K = size(Pk_x, 1);
epsilon = 1e-5;

Sigma = zeros(N,N, K);

for k = 1:K
    
    den = sum(Pk_x(k,:));
    Priors(1,k)=sum(Pk_x(k,:))/M;
    Mu(:,k)=sum(Pk_x(k,:).*X,2)./den;
    X_c = X-Mu(:,k);
    
    switch params.cov_type
        
    case "full"
        Sigma(:,:,k) = ((Pk_x(k,:).*X_c*X_c')./den) + epsilon;
        
    case "diag"
        Sigma(:,:,k) = ((Pk_x(k,:).*X_c*X_c')./den) + epsilon;
        Sigma(:,:,k) = diag(diag(Sigma(:,:,k)));
        
    case "iso"
        Sigma(:,:,k) = diag( sum(Pk_x(k,:).*sum(X_c.*X_c,1))./(N*den) + zeros(1,N) + epsilon);
    end
    
end



end

