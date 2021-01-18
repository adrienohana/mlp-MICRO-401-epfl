function [y_est, var_est] = gmr(Priors, Mu, Sigma, X, in, out)
%GMR This function performs Gaussian Mixture Regression (GMR), using the 
% parameters of a Gaussian Mixture Model (GMM) for a D-dimensional dataset,
% for D= N+P, where N is the dimensionality of the inputs and P the 
% dimensionality of the outputs.
%
% Inputs -----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM 
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the 
%              K GMM components.
%   o X:       N x M array representing M datapoints of N dimensions.
%   o in:      1 x N array representing the dimensions of the GMM parameters
%                to consider as inputs.
%   o out:     1 x P array representing the dimensions of the GMM parameters
%                to consider as outputs. 
% Outputs ----------------------------------------------------------------
%   o y_est:     P x M array representing the retrieved M datapoints of 
%                P dimensions, i.e. expected means.
%   o var_est:   P x P x M array representing the M expected covariance 
%                matrices retrieved. 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,K] = size(Priors);
[N,M] = size(X);
[~,P] = size(out);

prob = zeros(K,M);
den = 0;

for i=1:K
    prob(i,:) = gaussPDF(X, Mu(in,i), Sigma(in,in,i));
    den = den + Priors(1,i)*prob(i,:); 
end

Beta = zeros(K,M);
Mu_ = zeros(P,M,K);
Sigma_ = zeros(P,P,K);
y_est = zeros(P,M);
yy = N+1;

for i=1:K
    Beta(i,:) = Priors(1,i) * prob(i,:) ./ den;
    X_c = X-repmat(Mu(in,i),1,M);
    Mu_(:,:,i) = (Sigma(out,in,i) * pinv(Sigma(in,in,i)) * X_c) + repmat(Mu(out,i),1,M);
    y_est = y_est + repmat(Beta(i,:),1,P) .* Mu_(:,:,i);
    Sigma_(:,:,i) = Sigma(yy:end,yy:end,i) - Sigma(yy:end,1:N,i) * pinv(Sigma(1:N,1:N,i)) * Sigma(1:N,yy:end,i);
end

var_est = zeros(P,P,M);

for i = 1:M
    sum = zeros(1,P);
    for j = 1:K
        sum = sum + Beta(j,i) * Mu_(:,i,j);
        var_est(:,:,i) = var_est(:,:,i) + Beta(j,i) * (Mu_(:,i,j) * Mu_(:,i,j)' + Sigma_(:,:,j));
    end
    var_est(:,:,i) = var_est(:,:,i) - sum * sum';
end

end

