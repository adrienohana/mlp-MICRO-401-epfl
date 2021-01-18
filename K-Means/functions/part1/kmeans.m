function [labels, Mu, Mu_init, iter] =  kmeans(X,K,init,type,MaxIter,plot_iter)
%MY_KMEANS Implementation of the k-means algorithm
%   for clustering.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o K        : (int), chosen K clusters
%       o init     : (string), type of initialization {'sample','range'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%       o plot_iter: (bool), boolean to plot iterations or not (only works with 2d)
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a vector with predicted labels labels \in {1,..,k} 
%                   corresponding to the k-clusters for each points.
%       o Mu       : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N 
%       o Mu_init  : (N x k), same as above, corresponds to the centroids used
%                            to initialize the algorithm
%       o iter     : (int), iteration where algorithm stopped
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TEMPLATE CODE (DO NOT MODIFY)
% Auxiliary Variable
[D, N] = size(X);
d_i    = zeros(K,N);
k_i    = zeros(1,N);
r_i    = zeros(K,N);
if plot_iter == [];plot_iter = 0;end
tolerance = 1e-6;
MaxTolIter = 10;

% Output Variables
Mu     = zeros(D, K);
labels = zeros(1,N);


%% INSERT CODE HERE
Mu_init=kmeans_init(X, K, init);

%% TEMPLATE CODE (DO NOT MODIFY)
% Visualize Initial Centroids if N=2 and plot_iter active
colors     = hsv(K);
if (D==2 && plot_iter)
    options.title       = sprintf('Initial Mu with %s method', init);
    ml_plot_data(X',options); hold on;
    ml_plot_centroids(Mu_init',colors);
end

iter=0;
%% INSERT CODE HERE

% d --> (k,m) d(k,m) est la distance entre le datapoint m et le centroide k
%colonne de d == distances entre un data point et tous les centroides
% X --> (n,m)
% Mu--> (n,k)

% X--> (D, N)
% d--> (K, N)
% r--> (K ,N)

has_converged = false;
tol_iter = 0;
Mu = Mu_init;
while has_converged==false
    labels = zeros(1,N);
    r = zeros(K,N);
    iter=iter+1;
    d = distance_to_centroids(X, Mu, type);
    %pour chaque colonne (data point) i, trouver l'id du centroide avec la
    %plus petite distance. inscrire dans labels le numero du centroid
    for i=1:N
        [min_dist,min_id_k] = min(d(:,i));
        %sprintf('min_id_k is : ',min_id_k)
        %sprintf('min_dist is : ',min_dist)
        labels(:,i) = min_id_k;
        r(min_id_k,i) = 1;
    end
    %sprintf('range k : ',max(labels)
    Mu_previous=Mu;
    for j=1:K
        num = r(j,:)*X';
        den = sum(r(j,:));
        Mu(:,j) = num/den;
    end
    
     %if cluster is empty
     for i=1:K
         if r(i,:) == zeros(1,N)
              Mu_init =  kmeans_init(X, K, init);
             Mu = Mu_init;
             empty = 0;
             iter = 0;
             has_converged = false;
         end
     end
    [has_converged, tol_iter] = check_convergence(Mu, Mu_previous, iter, tol_iter, MaxIter, MaxTolIter, tolerance);
end
%% TEMPLATE CODE (DO NOT MODIFY)
if (D==2 && plot_iter)
    options.labels      = labels;
    options.class_names = {};
    options.title       = sprintf('Mu and labels after %d iter', iter);
    ml_plot_data(X',options); hold on;    
    ml_plot_centroids(Mu',colors);
end


end