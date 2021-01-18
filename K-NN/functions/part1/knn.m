function [ y_est ] =  knn(X_train,  y_train, X_test, params)
%MY_KNN Implementation of the k-nearest neighbor algorithm
%   for classification.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {1,2} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o params : struct array containing the parameters of the KNN (k, d_type)
%
%   output ----------------------------------------------------------------
%
%       o y_est   : (1 x M_test), a vector with estimated labels y \in {1,2} 
%                   corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N,M_train] = size(X_train);
[N,M_test] = size(X_test);
 
d = zeros(M_test,M_train);

for i=1:M_test
    for j=1:M_train
        d(i,j)= compute_distance(X_test(:,i),X_train(:,j),params);
    end
end

%sort distances by ascending order between lines
%line i contains ascending distances between all train samples
%and test sample i
[~,nn_ids] = sort(d, 2);

%keep k nearest neighbors in train for each test sample
nn_ids = nn_ids(:,1:params.k);

%for each train
nn_labels = y_train(nn_ids);

labels = unique(y_train);
nb_labels = size(labels,2);

temp = zeros(M_test,nb_labels);

for i=1:nb_labels
    temp(:,i)= sum(labels(i) == nn_labels,2);
end

[~,output_ids] = max(temp,[],2);

y_est = labels(output_ids);

if params.k == 1
    y_est = nn_labels;
end


end