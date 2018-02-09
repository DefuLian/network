%% testing multi-class classification by varying ratio of subspace learning
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
dim = 128;
n = length(network);

net = transform_network(network, 1, 1);
[U, S] = svds(net, dim);

B_svd = proj_hamming_balance(U * S);
num = 3;
loss = zeros(num,500);
for i=3:2:7
    [~,~,loss((i-1)/2,:)] = dne_ao_itq_subspace(net, B_svd, 'ratio',i*0.1);
end


dlmwrite(sprintf('%s/%s-dataset/data/loss_convergence_random.txt', data_dir_all, dataset), loss(1:3,:), 'precision', '%.5f');


%%

[~,~, ll_3] = dne_ao_itq_subspace(net, B_svd, 'ratio',3*0.1);
%%
ll2 = ll(ll>0);
ll1 = loss(1,:);
ll1=ll1(ll1>0);
ll3 = ll_3(ll_3>0);
ll4=loss(3,:);
ll4=ll4(ll4>0);
plot(ll1);hold on; plot(ll2);hold on; plot(ll3);hold on; plot(ll4)
