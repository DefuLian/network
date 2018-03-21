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

%%


%plot(loss_rand(loss_rand>0)); hold on
%plot(loss_init(loss_init>0));

loss_rand = zeros(500,4);
loss_init = zeros(500,4);
for i=[2,3,1,4]
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
Ts = [10,10,1,1];
%i=2;
data_dir_all = '/home/dlian/data/network/';
dataset = datasets{i};
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
T = Ts(i);
dim = 128;
n = length(network);

if T==1
    net = transform_network(network, T, 1);
    [U, S] = svds(net, dim);
    V = diag(1./diag(S)) * U.' * net;
else
    load(sprintf('%s/%s-dataset/data/netmf.mat', data_dir_all, dataset))
    S = embedding.'*embedding;
    U = embedding*diag(1./sqrt(diag(S)));
    V = diag(1./diag(S)) * U.' * net;
end

B_svd = sign(randn(n,dim));
[~,~,loss_rand(:,i)] = dne_ao_itq_subspace(net, B_svd);
B_svd = proj_hamming_balance(U * S);
[~,~,loss_init(:,i)] = dne_ao_itq_subspace(net, B_svd);
end
save('/home/dlian/data/network/loss_rand_init.mat', 'loss_rand', 'loss_init')
%%
i=4;
plot(loss_rand(loss_rand(:,i)>0,i)); hold on;
plot(loss_init(loss_init(:,i)>0,i)); 
