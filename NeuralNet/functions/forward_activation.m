function [A] = forward_activation(Z, Sigma)
%FORWARD_ACTIVATION Compute the value A of the activation function given Z
%   inputs:
%       o Z (NxM) Z value, input of the activation function. The size N
%       depends of the number of neurons at the considered layer but is
%       irrelevant here.
%       o Sigma (string) type of the activation to use
%
%   outputs:
%       o A (NXM) value of the activation function


switch Sigma
    
    case 'sigmoid'
        A  = 1./(1+exp(-Z));
        
    case 'tanh'
        A = tanh(Z);
        
    case 'relu'
        A = max(0,Z);
        
    case  'leakyrelu'
        A = max(0.01*Z,Z);
        
    case 'softmax'
        A= exp(Z-max(Z)) ./ sum(exp(Z-max(Z)));
end

end

