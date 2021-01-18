function [W, W0] = initialize_weights(LayerSizes, type)
%INITIALIZE_WEIGHTS Initialize the wieghts of the network according to the
%desired type of initialization
%   inputs:
%       o LayerSizes{1x(L+1)} Cell array containing the sizes of each layers.
%       Also contains the size of A0 input layer
%       o type (string) type of the desired initialization ('random' or 'zeros')
%
%   outputs:
%       o W {Lx1} cell array containing the weight matrices for all the layers 
%       o W0 {Lx1} cell array containing the bias matrices for all the layers

[~,L] = size(LayerSizes);
L = L-1;
%{ [n0] [n1] [n2].... }

W = cell(L,1);
W0 = cell(L,1);

%{ [w01 w02 w03 ...] [w11 w12 w13 ..... ] .... [wL1 wL2 wL3 ..... ]}
switch type
    
    case 'zeros'
        for c = 1:L
            nb_neurons_i = LayerSizes{c};
            nb_neurons_j = LayerSizes{c+1};
            W{c} = zeros([nb_neurons_j, nb_neurons_i]);
            W0{c} = zeros([nb_neurons_j 1]);
        end
        
    case 'random'
        for c = 1:L
            nb_neurons_i = LayerSizes{c};
            nb_neurons_j = LayerSizes{c+1};
            W{c} = randn([nb_neurons_j, nb_neurons_i]);
            W0{c} = randn([nb_neurons_j 1]);
        end
        
end

end

