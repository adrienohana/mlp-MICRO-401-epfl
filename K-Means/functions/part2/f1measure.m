function [F1_overall, P, R, F1] =  f1measure(cluster_labels, class_labels)
%MY_F1MEASURE Computes the f1-measure for semi-supervised clustering
%
%   input -----------------------------------------------------------------
%   
%       o class_labels     : (1 x M),  M-dimensional vector with true class
%                                       labels for each data point
%       o cluster_labels   : (1 x M),  M-dimensional vector with predicted 
%                                       cluster labels for each data point
%   output ----------------------------------------------------------------
%
%       o F1_overall      : (1 x 1)     f1-measure for the clustered labels
%       o P               : (nClusters x nClasses)  Precision values
%       o R               : (nClusters x nClasses)  Recall values
%       o F1              : (nClusters x nClasses)  F1 values
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clusters = unique(cluster_labels);
classes = unique(class_labels);

n_clusters = max(clusters);
n_classes = max(classes);

n_ik = zeros(n_clusters,n_classes);

for k=1:n_clusters
    for i=1:n_classes
            %inter(i,j) is the number of sample points attributed to
            %cluster i AND class j
            n_ik(k,i)= sum(cluster_labels==clusters(k) & class_labels==classes(i));
    end
end

%no of elements per class
c_i = sum(n_ik,1);

%no of elements per cluster
k_ = sum(n_ik,2);

R = n_ik./repmat(c_i,n_clusters,1);
P = n_ik./repmat(k_,1,n_classes);

F1 = 2*R.*P./(R+P);

F1_overall = sum( c_i.*max(F1,[],1) )/length(class_labels);

F1(isnan(F1)) = 0;


end
