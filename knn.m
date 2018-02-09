datasets={'BlogCatalog','PPI','Wiki','Flickr'};
data_dir_all = '/home/dlian/Data/data/network/';
time=zeros(4,2);
for i=1:4
dataset = datasets{i};
load(sprintf('%s/%s-dataset/data/node_rec/hash_code.mat', data_dir_all, dataset))
%save(sprintf('%s/%s-dataset/data/node_rec/code.mat', data_dir_all, dataset), 'B','Q','-v7.3');
load(sprintf('%s/%s-dataset/data/node_rec/netmf_embed.mat', data_dir_all, dataset))
time(i,1) = knn_search_exp(W_itq_svd,W_itq_svd,200);
time(i,2) = knn_search_exp(emb_netmf_o,emb_netmf_o,200);
end
%dist = han_dist(W_itq_svd(1:100,:), W_itq_svd);
%dist2 = hammingDist(W_itq_svd(1:100,:), W_itq_svd);
dlmwrite(sprintf('%s/knn_ranking_comparison.txt', data_dir_all), time);

mex -largeArrayDims han_dist.cpp
mex -largeArrayDims mult.cpp
