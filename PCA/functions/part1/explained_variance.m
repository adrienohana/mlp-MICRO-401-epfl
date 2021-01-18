function [ExpVar, CumVar, p_opt] = explained_variance(EigenValues, var_threshold)
%EXPLAINED_VARIANCE Function that returns the optimal p given a desired
%   explained variance.
%
%   input -----------------------------------------------------------------
%   
%       o EigenValues     : (N x 1), Diagonal Matrix composed of lambda_i 
%       o var_threshold   : Desired Variance to be explained
%  
%   output ----------------------------------------------------------------
%
%       o ExpVar  : (N x 1) vector of explained variance
%       o CumVar  : (N x 1) vector of cumulative explained variance
%       o p_opt   : optimal principal components given desired Var

%calculate explained variance vector
ExpVar = EigenValues/sum(EigenValues);
CumVar = cumsum(ExpVar);

%find optimal p (no. of principal components) given var_threshold
p=1;
while (CumVar(p) < var_threshold) & (p <= length(EigenValues))
    p=p+1;
end
p_opt=p;
end

