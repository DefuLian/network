for i=[2,3,1,4]
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
Ts = [10,10,1,1];
%i=2;
data_dir_all = '/home/dlian/Data/data/network/';
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


B_svd = proj_hamming_balance(U * S);
num = 10;
timing = zeros(num,1);
data_dir = sprintf('%s/%s-dataset/data/mc_ratio', data_dir_all, dataset);
for g=1:num
    tic;B_itq_svd = dne_mf(net, B_svd, 'ratio',g*0.1);timing(g)=toc;
    file_name = sprintf('%s/embedding_%s_%d_fixiter.txt', data_dir, 'bitq_svd', g);
    fileid = fopen(file_name, 'w');
    fprintf(fileid, '%d %d\n', n, dim);
    fclose(fileid);
    dlmwrite(file_name, [(0:n-1)', B_itq_svd], 'delimiter', ' ', '-append')
end
dlmwrite(sprintf('%s/timing2.txt', data_dir), timing);
end


%% parameter sensivity for generating data
rng(1000)
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
train10 = network;
[train9, ~]=split_network(network, 'un', 0.9);
[train8, ~]=split_network(network, 'un', 0.8);
[train7, ~]=split_network(network, 'un', 0.7);
[train6, ~]=split_network(network, 'un', 0.6);
[train5, ~]=split_network(network, 'un', 0.5);
[train4, ~]=split_network(network, 'un', 0.4);
[train3, ~]=split_network(network, 'un', 0.3);
[train2, ~]=split_network(network, 'un', 0.2);
[train1, ~]=split_network(network, 'un', 0.1);
save(sprintf('%s/%s-dataset/data/mc_parameter_sensitivity/%s.mat', data_dir_all, dataset, dataset), ...
    'train10', 'train9', 'train8','train7','train6','train5','train4','train3','train2','train1');

clear
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
load(sprintf('%s/%s-dataset/data/node_rec/node_rec0/%s.mat', data_dir_all, dataset, dataset))
network = train;
train10 = train;
[train9, ~]=split_network(network, 'un', 0.9);
[train8, ~]=split_network(network, 'un', 0.8);
[train7, ~]=split_network(network, 'un', 0.7);
[train6, ~]=split_network(network, 'un', 0.6);
[train5, ~]=split_network(network, 'un', 0.5);
[train4, ~]=split_network(network, 'un', 0.4);
[train3, ~]=split_network(network, 'un', 0.3);
[train2, ~]=split_network(network, 'un', 0.2);
[train1, ~]=split_network(network, 'un', 0.1);
save(sprintf('%s/%s-dataset/data/node_rec/node_rec0/nr_parameter_sensitivity/%s.mat', data_dir_all, dataset, dataset), ...
    'train10', 'train9', 'train8','train7','train6','train5','train4','train3','train2','train1');

%% generating hashing for varying dim
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
net = transform_network(network, 1, 1);
[U, S] = svds(net, 512);
V = diag(1./diag(S)) * U.' * net;
B_svd = proj_hamming_balance(U * S);
n = length(network);
data_dir = sprintf('%s/%s-dataset/data/mc_parameter_sensitivity/', data_dir_all, dataset);
for dim = 512 %[16,32,64,128,256];
B_itq_svd = dne_ao_itq(net, B_svd(:,1:dim), 0.);
file_name = sprintf('%s/embedding_%d.txt', data_dir, dim);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_itq_svd], 'delimiter', ' ', '-append')
end
%% generate hashing for varying ratio
clear
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
data_dir = sprintf('%s/%s-dataset/data/mc_parameter_sensitivity/', data_dir_all, dataset);
load(sprintf('%s/%s.mat', data_dir, dataset))
dim = 128;
for i=1:10
    eval(['network=', sprintf('train%d',i),';']);
    n = length(network);
    net = transform_network(network, 1, 1);
    [U, S] = svds(net, dim);
    B_svd = proj_hamming_balance(U * S);
    B_itq_svd = dne_ao_itq(net, B_svd, 0.);
    file_name = sprintf('%s/embedding_ratio_%d.txt', data_dir, i);
    fileid = fopen(file_name, 'w');
    fprintf(fileid, '%d %d\n', n, dim);
    fclose(fileid);
    dlmwrite(file_name, [(0:n-1)', B_itq_svd], 'delimiter', ' ', '-append')
end

%% mc for dne

dataset = 'Flickr';
data_dir_all = sprintf('/home/dlian/Data/data/network/%s-dataset/data/node_rec/node_rec0/', dataset);
load(sprintf('%s/%s.mat', data_dir_all, dataset))
network=train;
net = transform_network(network, 1, 1);
[U, S] = svds(net, 512);
V = diag(1./diag(S)) * U.' * net;
B_svd = proj_hamming_balance(U * S);
n = length(network);
result = cell(5,1);
i=1;
for dim = 512 %[16,32,64,128,256]
B_itq_svd = dne_ao_itq(net, B_svd(:,1:dim), 0.);
P=B_itq_svd;
Q=B_itq_svd;
result{i} = evaluate_item(train, test, P, Q, -1, 200);
i=i+1;
end
result = cell2mat(result);
ndcg = cell2mat({result.ndcg}');
ndcg = ndcg(:,50);
auc = cell2mat({result.auc}');
mpr = cell2mat({result.mpr}');
final = full([ndcg, auc, mpr]);

dlmwrite(sprintf('%s/nr_dim_metric.txt', data_dir_all), final, '-append', 'precision','%.4f');


%% mc for netmf
dataset = 'Flickr';
data_dir_all = sprintf('/home/dlian/Data/data/network/%s-dataset/data/node_rec/node_rec0/', dataset);
load(sprintf('%s/%s.mat', data_dir_all, dataset))
result = cell(6,1);
i=1;
for dim = [16,32,64,128,256,512]
load(sprintf('%s/nr_parameter_sensitivity/netmf_embed_dim_%d.mat',data_dir_all, dim))
P=emb_netmf_o;
Q=emb_netmf_o;
result{i} = evaluate_item(train, test, P, Q, -1, 200);
i=i+1;
end
result = cell2mat(result);
ndcg = cell2mat({result.ndcg}');
ndcg = ndcg(:,50);
auc = cell2mat({result.auc}');
mpr = cell2mat({result.mpr}');
final = full([ndcg, auc, mpr]);

dlmwrite(sprintf('%s/nr_dim_metric_netmf.txt', data_dir_all), final, 'precision','%.4f');

%% nr for dne
clear
dataset = 'Flickr';
data_dir_all = sprintf('/home/dlian/Data/data/network/%s-dataset/data/node_rec/node_rec0/', dataset);
load(sprintf('%s/%s.mat',data_dir_all, dataset));
data_dir = sprintf('%s/nr_parameter_sensitivity/', data_dir_all);
load(sprintf('%s/%s.mat', data_dir, dataset))
dim = 128;
result = cell(10,1);
for i=1:10
    eval(['network=', sprintf('train%d',i),';']);
    n = length(network);
    net = transform_network(network, 1, 1);
    [U, S] = svds(net, dim);
    B_svd = proj_hamming_balance(U * S);
    B_itq_svd = dne_ao_itq(net, B_svd, 0.);
    P=B_itq_svd;
    Q=B_itq_svd;
    result{i} = evaluate_item(train, test, P, Q, -1, 200);
end

result = cell2mat(result);
ndcg = cell2mat({result.ndcg}');
ndcg = ndcg(:,50);
auc = cell2mat({result.auc}');
mpr = cell2mat({result.mpr}');
final = full([ndcg, auc, mpr]);

dlmwrite(sprintf('%s/nr_ratio_metric.txt', data_dir_all), final, 'precision','%.4f');

%% netmf for netmf
dataset = 'Flickr';
data_dir_all = sprintf('/home/dlian/Data/data/network/%s-dataset/data/node_rec/node_rec0/', dataset);
load(sprintf('%s/%s.mat',data_dir_all, dataset));
result = cell(10,1);
for i=7:10
    load(sprintf('%s/nr_parameter_sensitivity/netmf_embed_ratio_%d.mat',data_dir_all, i))
    P=emb_netmf_o;
    Q=emb_netmf_o;
    result{i} = evaluate_item(train, test, P, Q, -1, 200);
end

result = cell2mat(result(7:10));
ndcg = cell2mat({result.ndcg}');
ndcg = ndcg(:,50);
auc = cell2mat({result.auc}');
mpr = cell2mat({result.mpr}');
final = full([ndcg, auc, mpr]);

dlmwrite(sprintf('%s/nr_ratio_metric_netmf.txt', data_dir_all), final, 'precision','%.4f');

%% timing when dimension k changes
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
time = zeros(6,4);
i=1;
for dim=[16,32,64,128,256,512]
    time(i,:) = testing_time(network, dim);
    i=i+1;
end
dlmwrite(sprintf('%s/%s-dataset/data/timing_dim.txt', data_dir_all, dataset), time);

clear
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
data_dir = sprintf('%s/%s-dataset/data/mc_parameter_sensitivity/', data_dir_all, dataset);
load(sprintf('%s/%s.mat', data_dir, dataset))
dim = 128;
time=zeros(10,4);
for i=1:10
    eval(['network=', sprintf('train%d',i),';']);
    time(i,:) = testing_time(network, dim);
end
dlmwrite(sprintf('%s/%s-dataset/data/timing_ratio.txt', data_dir_all, dataset), time);

%%
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'PPI';
data = cell(10,1);
for i=0:9
data{i+1} = dlmread(sprintf('%s/%s-dataset/data/node_rec/node_rec%d/nr_result_%s.txt',data_dir_all, dataset, i, dataset));
end
result = data{1};
for i=2:10
    result = result + data{i};
end
result = result/10;
dlmwrite(sprintf('%s/%s-dataset/data/node_rec/nr_result_%s.txt', data_dir_all, dataset, dataset),...
    result, 'precision','%.4f');    