function [MSE, NMSE, Rsquared] = regression_metrics( yest, y )
%REGRESSION_METRICS Computes the metrics (MSE, NMSE, R squared) for 
%   regression evaluation
%
%   input -----------------------------------------------------------------
%   
%       o yest  : (P x M), representing the estimated outputs of P-dimension
%       of the regressor corresponding to the M points of the dataset
%       o y     : (P x M), representing the M continuous labels of the M 
%       points. Each label has P dimensions.
%
%   output ----------------------------------------------------------------
%
%       o MSE       : (1 x 1), Mean Squared Error
%       o NMSE      : (1 x 1), Normalized Mean Squared Error
%       o R squared : (1 x 1), Coefficent of determination
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[P,M] = size(y);
MSE = (1/M) * sum(norm((yest-y))^2);

Mu = (1/M) * sum(y,2);
Var = (1/(M-1)) * sum(norm(y-repmat(Mu,1,P))^2);
NMSE = MSE / Var;

yest_avg = mean(yest,2);
y_avg = mean(y,2);
num = norm(sum( (y - repmat(y_avg,1,P)) .* (yest - repmat(yest_avg,1,P)) ) )^2;
den = sum(norm( y - repmat(yest_avg,1,P))^2) * sum(norm(yest - repmat(yest_avg,1,P))^2);

Rsquared = num/den;



end

