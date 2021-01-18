function [metrics] = cross_validation_gmr( X, y, F_fold, valid_ratio, k_range, params )
%CROSS_VALIDATION_GMR Implementation of F-fold cross-validation for regression algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (P x M) array representing the y vector assigned to
%                           each datapoints
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o valid_ratio  : (double), Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%       o params    : parameter strcuture of the GMM
%
%   output ----------------------------------------------------------------
%       o metrics : (structure) contains the following elements:
%           - mean_MSE   : (1 x K), Mean Squared Error computed for each value of k averaged over the number of folds.
%           - mean_NMSE  : (1 x K), Normalized Mean Squared Error computed for each value of k averaged over the number of folds.
%           - mean_R2    : (1 x K), Coefficient of Determination computed for each value of k averaged over the number of folds.
%           - mean_AIC   : (1 x K), Mean AIC Scores computed for each value of k averaged over the number of folds.
%           - mean_BIC   : (1 x K), Mean BIC Scores computed for each value of k averaged over the number of folds.
%           - std_MSE    : (1 x K), Standard Deviation of Mean Squared Error computed for each value of k.
%           - std_NMSE   : (1 x K), Standard Deviation of Normalized Mean Squared Error computed for each value of k.
%           - std_R2     : (1 x K), Standard Deviation of Coefficient of Determination computed for each value of k averaged over the number of folds.
%           - std_AIC    : (1 x K), Standard Deviation of AIC Scores computed for each value of k averaged over the number of folds.
%           - std_BIC    : (1 x K), Standard Deviation of BIC Scores computed for each value of k averaged over the number of folds.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,~] = size(X);
[P,~] = size(y);
[~,K] = size(k_range)

in  = [1:N];
out = [(N+1):(N+P)];

for i=1:K
    params.k = i;
    
    for j=1:F_fold
        [X_train, y_train, X_test, y_test] = split_regression_data(X, y, valid_ratio);
        trainset = [X_train;y_train];
        [Priors, Mu, Sigma, ~] = gmmEM(trainset, params);
        [y_est, ~] = gmr(Priors, Mu, Sigma, X_test, in, out);
        [MSE_list(j), NMSE_list(j), Rsquared_list(j)] = regression_metrics(y_est, y_test);
        [AIC_list(j), BIC_list(j)] =  gmm_metrics(trainset, Priors, Mu, Sigma, params.cov_type);
    end
    
    metrics.mean_MSE(1,i) = mean(MSE_list);
    metrics.mean_NMSE(1,i) = mean(NMSE_list);
    metrics.mean_R2(1,i) = mean(Rsquared_list);
    metrics.mean_AIC(1,i) = mean(AIC_list);
    metrics.mean_BIC(1,i) = mean(BIC_list);
    metrics.std_MSE(1,i) = std(MSE_list);
    metrics.std_NMSE(1,i) = std(NMSE_list);
    metrics.std_R2(1,i) = std(Rsquared_list);
    metrics.std_AIC(1,i) = std(AIC_list);
    metrics.std_BIC(1,i) = std(BIC_list);
end


end

