function [dZ] = backward_activation(Z, Sigma)
%BACKWARD_ACTIVATION Compute the derivative of the activation function
%evaluated in Z
%   inputs:
%       o Z (NxM) Z value, input of the activation function. The size N
%       depends of the number of neurons at the considered layer but is
%       irrelevant here.
%       o Sigma (string) type of the activation to use
%   outputs:
%       o dZ (NXM) derivative of the activation function


switch Sigma
    
    case 'sigmoid'
        A  = (1./(1+exp(-Z))).*(1-(1./(1+exp(-Z))));
        
    case 'tanh'
        A = 1 - tanh(Z).*tanh(Z);
        
    case 'relu'
        A = max(0,Z);
        A(A~=0)=1;
   
    case  'leakyrelu'
        A = max(0,Z);
        A(A~=0)=1;
        A(A==0)=0.01;
        
end

dZ = A;

end

