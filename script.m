D = sum(network)+1e-10;
D = spdiags(1./D.', 0, length(network), length(network));
network_ = D * network * D;
[I, J, K] = find(network_);
b = 5;
K = log(K .* nnz(network)./b);
ind = K>0;
network_1 = sparse(I(ind), J(ind), K(ind), length(network), length(network));
%network_1 = sparse(I, J, K, length(network), length(network));
[P,S,V] = svds(network_1, 128);
PP = P * diag(diag(S));
%[P,Q] = iccf(network_1, 'alpha',10, 'max_iter', 50, 'K', 128);
%PP = P(:,1:128);
%PP = NormalizeFea(PP, 'row', 1);
P1 = [(1:size(P,1))',PP];
fileid = fopen('/home/dlian/data/network/BlogCatalog-dataset/data/wals_embedding.txt', 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite('/home/dlian/data/network/BlogCatalog-dataset/data/wals_embedding.txt', P1, 'delimiter', ' ', '-append')
%%
dataset='PPI';
data_dir_all = '/home/dlian/Data/data/network/';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
[J,I,K]=find(network.');
ind=(I<=J);
data = [I,J];
data_uniq = data(ind,:);
dlmwrite(sprintf('%s/%s-dataset/data/edges.csv', data_dir_all, dataset), data_uniq, 'delimiter',',');
[J,I,~]=find(group.');
data=[I,J];
dlmwrite(sprintf('%s/%s-dataset/data/group-edges.csv', data_dir_all, dataset), data, 'delimiter',',');
%%
dataset = 'BlogCatalog';
data_dir = '/home/dlian/Data/data/network/';
num_groups = max(dlmread(sprintf('%s/%s-dataset/data/groups.csv', data_dir, dataset), ','));
num_nodes = max(dlmread(sprintf('%s/%s-dataset/data/nodes.csv', data_dir, dataset), ','));
network = readContent(sprintf('%s/%s-dataset/data/edges.csv', data_dir, dataset), 'sep', ',', 'zero_start', false, 'nrows', num_nodes, 'ncols', num_nodes);
network = network + network.';
group = readContent(sprintf('%s/%s-dataset/data/group-edges.csv', data_dir, dataset), 'sep', ',', 'zero_start', false, 'nrows', num_nodes, 'ncols', num_groups);
save(sprintf('%s/%s-dataset/data/%s.mat', data_dir, dataset, dataset), 'network', 'group')

dataset = 'BlogCatalog';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir, dataset, dataset))
mat = transform_network(network, 1, 5);
sum(sum(mat))
nnz(mat)
%%
[val,ind]=maxk(sum(group),3);
group_filter = group(:,ind);
row_ind = sum(group_filter,2)>0;
label = num2cell(group_filter(row_ind,:),2);
label=cellfun(@find, label, 'UniformOutput',false);
label = cellfun(@(x) x(1), label);
gscatter(mappedX(row_ind,1), mappedX(row_ind,2), label);
%%
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'BlogCatalog';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
T = 1;
b = 1;
dim = 128;
n = length(network);

if T==1
    net = transform_network(network, T, b);
    [U, S] = svds(net, dim);
    V = diag(1./diag(S)) * U.' * net;
else
    if strcmp(dataset, 'Flickr')
        load(sprintf('%s/%s-dataset/data/netmf_lr_1024.mat', data_dir_all, dataset))
        U = net_lr(:,end-dim+1:end);
        S = U.'*U;
        V = U.';
        net = net_lr;
    else
        load(sprintf('%s/%s-dataset/data/netmf.mat', data_dir_all, dataset))
        S = embedding.'*embedding;
        U = embedding*diag(1./sqrt(diag(S)));
        V = diag(1./diag(S)) * U.' * net;
    end
end


B_svd = proj_hamming_balance(U * S);
B_itq_svd = dne_ao_itq(net, B_svd, 0.);

[P,Q]=mf_itq(net,dim);
B_mf = proj_hamming_balance(P);
B_itq_mf = dne_ao_itq(net, B_mf, 0.);

%[P, Q] = initialize_mf(net, dim);
%B_mf = dne_ao(net, B_svd);


data_dir = sprintf('%s/%s-dataset/data/', data_dir_all, dataset);

file_name = sprintf('%s/embedding_%s_%d_%d.txt', data_dir, 'bsvd', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_svd], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s_%d_%d.txt', data_dir, 'bmf', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_mf], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s_%d_%d.txt', data_dir, 'bitq_svd', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_itq_svd], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s_%d_%d.txt', data_dir, 'bitq_mf', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_itq_mf], 'delimiter', ' ', '-append')

%file_name = sprintf('%s/embedding_%s_%d.txt', data_dir, 'bmf0', T);
%fileid = fopen(file_name, 'w');
%fprintf(fileid, '%d %d\n', n, dim);
%fclose(fileid);
%dlmwrite(file_name, [(0:n-1)', proj_hamming_balance(P)], 'delimiter', ' ', '-append')

%file_name = sprintf('%s/embedding_%s_%d.txt', data_dir, 'bmf', T);
%fileid = fopen(file_name, 'w');
%fprintf(fileid, '%d %d\n', n, dim);
%fclose(fileid);
%dlmwrite(file_name, [(0:n-1)', B_mf], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s_%d_%d.txt', data_dir, 'mf', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', NormalizeFea(P, 'row', 1)], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s_%d_%d_nn.txt', data_dir, 'mf', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', P], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s_%d_%d.txt', data_dir, 'svd', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', NormalizeFea(U * S, 'row', 1)], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s_%d_%d_nn.txt', data_dir, 'svd', T,b);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', U*S], 'delimiter', ' ', '-append')

%file_name = sprintf('%s/baseline/embedding_%s_%d_%d_nn.txt', data_dir, 'svd', T,b);
%fileid = fopen(file_name, 'w');
%fprintf(fileid, '%d %d\n', n, dim);
%fclose(fileid);
%dlmwrite(file_name, [(0:n-1)', U*S], 'delimiter', ' ', '-append')
%

%file_name = sprintf('%s/embedding_%s_%d.txt', data_dir, 'mf', T);
%fileid = fopen(file_name, 'w');
%fprintf(fileid, '%d %d\n', n, dim);
%fclose(fileid);
%dlmwrite(file_name, [(0:n-1)', NormalizeFea(P, 'row', 1)], 'delimiter', ' ', '-append')



%%
count = 0;
for gamma=[0, 0.01, 0.1, 1, 10]
    count = count + 1;
    B_itq = dne_ao_itq(net, B_svd, gamma);
    file_name = sprintf('%s/itq/embedding_%s_%d_%d.txt', data_dir, 'bitq', T, count);
    fileid = fopen(file_name, 'w');
    fprintf(fileid, '%d %d\n', n, dim);
    fclose(fileid);
    dlmwrite(file_name, [(0:n-1)', B_itq], 'delimiter', ' ', '-append')
end
%%
[B,Q]=dne_wao(net);
[train,test]=split_matrix(net, 'en',0.99);
metric_func = @(metric) metric.auc(1,1);
alg = @(varargin) item_recommend(@(mat) piccf(mat, 'K', 128, 'max_iter', 20, varargin{:}),...
    train, 'test', test);
[para,~, metric] = hyperp_search(alg, metric_func, 'alpha', [0,0.1,0.5,1,2]);

count = 0;
for alpha=[0, 1, 5, 10, 20]
    count = count + 1;
    [P,Q]=piccf(net, 'K', 128, 'max_iter', 20, 'alpha', 5);
    P = P(:,1:128);
    file_name = sprintf('%s/itq/embedding_%s_%d_%d.txt', data_dir, 'bwals', T, count);
    fileid = fopen(file_name, 'w');
    fprintf(fileid, '%d %d\n', n, dim);
    fclose(fileid);
    %dlmwrite(file_name, [(0:n-1)', NormalizeFea(P,'row',1)], 'delimiter', ' ', '-append')
    dlmwrite(file_name, [(0:n-1)', proj_hamming_balance(P)], 'delimiter', ' ', '-append')
end
count=2;
[B,Q]=dne_wao(net);
file_name = sprintf('%s/itq/embedding_%s_%d_%d.txt', data_dir, 'bqp', T, count);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
%dlmwrite(file_name, [(0:n-1)', NormalizeFea(P,'row',1)], 'delimiter', ' ', '-append')
dlmwrite(file_name, [(0:n-1)', B], 'delimiter', ' ', '-append')


[B,Q]=dne_ao_itq(net,B_svd,0.05);
[B,Q]=dne_ao_itq_subspace(net,B_svd,'gamma',0.05);
count=5;
file_name = sprintf('%s/itq/embedding_%s_%d_%d.txt', data_dir, 'itqp', T, count);
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
%dlmwrite(file_name, [(0:n-1)', NormalizeFea(P,'row',1)], 'delimiter', ' ', '-append')
dlmwrite(file_name, [(0:n-1)', B_mf], 'delimiter', ' ', '-append')


%%
data_dir_all = '/home/dlian/data/network/';
dataset = 'Flickr';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
dim = 128;
n = length(network);

if ~strcmp(dataset, 'Flickr')
    network_dense = full(network);
    SHparam.nbits = dim; % number of bits to code each sample
    SHparam = trainSH(network_dense, SHparam);
    [B_sh,U_sh] = compressSH(network_dense, SHparam);

    B_itq = compressITQ(network_dense,dim); 
    B_itq = B_itq * 2 - 1;
else
    SHparam.nbits = dim; % number of bits to code each sample
    SHparam = trainSH(network, SHparam);
    [B_sh,U_sh] = compressSH(network, SHparam);

    B_itq = compressITQ(network,dim); 
    B_itq = B_itq * 2 - 1;
end


[~,anchor]=litekmeans(network,1000);
anchor = sparse(anchor);
s = 5;
B_AGH_1 = OneLayerAGH_Train(network, anchor, dim, s, 0);
B_AGH_1=B_AGH_1*2-1;
B_AGH_2 = TwoLayerAGH_Train(network, anchor, dim, s, 0);
B_AGH_2=B_AGH_2*2-1;

method = 'IMH-LE';
options = InitOpt(method);
options.nbits = dim;
options.maxbits = dim;
[Embedding,Z_RS,~] = InducH(anchor, network, options);
EmbeddingX = Z_RS*Embedding;
B_IMH = (EmbeddingX > 0)*2-1;


data_dir = sprintf('%s/%s-dataset/data/baseline', data_dir_all, dataset);

file_name = sprintf('%s/embedding_%s.txt', data_dir, 'sh');
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_sh], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s.txt', data_dir, 'itq');
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_itq], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s.txt', data_dir, 'agh_1');
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_AGH_1], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s.txt', data_dir, 'agh_2');
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_AGH_2], 'delimiter', ' ', '-append')

file_name = sprintf('%s/embedding_%s.txt', data_dir, 'imh');
fileid = fopen(file_name, 'w');
fprintf(fileid, '%d %d\n', n, dim);
fclose(fileid);
dlmwrite(file_name, [(0:n-1)', B_IMH], 'delimiter', ' ', '-append')


%%
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Wiki';
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
dim = 128;
n = length(network);
rng(10)
[J,I,K]=find(network.');
ind=I<J;
asys_network=sparse(I(ind),J(ind),K(ind),n,n);
[asys_train, asys_test]=split_matrix(asys_network, 'un', 0.9);
train = asys_train + asys_train.';
test = asys_test + asys_test.';
save(sprintf('%s/%s-dataset/data/node_rec/%s.mat', data_dir_all, dataset, dataset), 'train', 'test')

[J,I,K]=find(train.');
ind=(I<=J);
data = [I,J];
data_uniq = data(ind,:);
dlmwrite(sprintf('%s/%s-dataset/data/node_rec/edges.csv', data_dir_all, dataset), data_uniq, 'delimiter',',');
%%
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
Ts = [10,10,1,1];
i=4;
data_dir_all = '/home/dlian/Data/data/network/';
dataset = datasets{i};
load(sprintf('%s/%s-dataset/data/node_rec/%s.mat', data_dir_all, dataset, dataset))
network=train;
T = Ts(i);
b = 1;
dim = 128;
n = length(network);

if T==1
    net = transform_network(network, T, b);
    [U, S] = svds(net, dim);
    V = diag(1./diag(S)) * U.' * net;
else
    load(sprintf('%s/%s-dataset/data/node_rec/netmf.mat', data_dir_all, dataset))
    S = embedding.'*embedding;
    U = embedding*diag(1./sqrt(diag(S)));
    V = diag(1./diag(S)) * U.' * net;
end


B_svd = proj_hamming_balance(U * S);
B_svd = (B_svd>0)*2-1;
W_svd = compactbit(B_svd>0);
B_itq_svd = dne_ao_itq(net, B_svd, 0.);
B_itq_svd = (B_itq_svd>0)*2-1;
W_itq_svd = compactbit(B_itq_svd>0);

if ~strcmp(dataset, 'Flickr')
    network_dense = full(network);
    SHparam.nbits = dim; % number of bits to code each sample
    SHparam = trainSH(network_dense, SHparam);
    [B_sh,U_sh] = compressSH(network_dense, SHparam);
    W_sh = compactbit(B_sh>0);
    
    B_itq = compressITQ(network_dense,dim); 
    W_itq = compactbit(B_itq);
    B_itq = B_itq * 2 - 1;
else
    SHparam.nbits = dim; % number of bits to code each sample
    SHparam = trainSH(network, SHparam);
    [B_sh,U_sh] = compressSH(network, SHparam);
    W_sh = compactbit(B_sh>0);
    
    B_itq = compressITQ(network,dim); 
    W_itq = compactbit(B_itq);
    B_itq = B_itq * 2 - 1;
end


[~,anchor]=litekmeans(network,1000);
anchor = sparse(anchor);
s = 5;
B_AGH_1 = OneLayerAGH_Train(network, anchor, dim, s, 0);
W_AGH_1 = compactbit(B_AGH_1);
B_AGH_1 = B_AGH_1*2-1;
B_AGH_2 = TwoLayerAGH_Train(network, anchor, dim, s, 0);
W_AGH_2 = compactbit(B_AGH_2);
B_AGH_2 = B_AGH_2*2-1;

method = 'IMH-LE';
options = InitOpt(method);
options.nbits = dim;
options.maxbits = dim;
[Embedding,Z_RS,~] = InducH(anchor, network, options);
EmbeddingX = Z_RS*Embedding;
B_IMH = (EmbeddingX > 0);
W_IMH = compactbit(B_IMH);
B_IMH = B_IMH*2-1;

[min(B_AGH_1(:)),min(B_AGH_2(:)),min(B_IMH(:)),min(B_itq(:)),min(B_itq_svd(:)),min(B_svd(:)),min(B_sh(:))]
save(sprintf('%s/%s-dataset/data/node_rec/hash_code.mat', data_dir_all, dataset), ...
    'W_AGH_1','W_AGH_2','W_IMH','W_itq','W_itq_svd','W_svd','W_sh',...
    'B_AGH_1','B_AGH_2','B_IMH','B_itq','B_itq_svd','B_svd','B_sh');

%% node recommendation
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
i=4;
dataset = datasets{i};
data_dir_all = '/home/dlian/data/network/';
load(sprintf('%s/%s-dataset/data/node_rec/hash_code.mat', data_dir_all, dataset))
load(sprintf('%s/%s-dataset/data/node_rec/%s.mat', data_dir_all, dataset, dataset))
n = size(train,1);
dim=128;
P=B_itq;
Q=B_itq;
result_itq = evaluate_item(train, test, P, Q, -1, 200);
P=B_itq_svd;
Q=B_itq_svd;
result_itq_svd = evaluate_item(train, test, P, Q, -1, 200);
P=B_svd;
Q=B_svd;
result_svd = evaluate_item(train, test, P, Q, -1, 200);
P=B_sh;
Q=B_sh;
result_sh = evaluate_item(train, test, P, Q, -1, 200);
P=B_AGH_1;
Q=B_AGH_1;
result_agh_1 = evaluate_item(train, test, P, Q, -1, 200);
P=B_AGH_2;
Q=B_AGH_1;
result_agh_2 = evaluate_item(train, test, P, Q, -1, 200);
P=B_IMH;
Q=B_IMH;
result_imh = evaluate_item(train, test, P, Q, -1, 200);
load(sprintf('%s/%s-dataset/data/node_rec/netmf_embed.mat', data_dir_all, dataset))
P = emb_netmf_o;
Q = emb_netmf_o;
result_netmf = evaluate_item(train, test, P, Q, -1, 200);
emb_deepwalk=load_embed(sprintf('%s/%s-dataset/data/node_rec/deepwalk_embed.txt', data_dir_all, dataset));
P = emb_deepwalk;
Q = emb_deepwalk;
result_deepwalk = evaluate_item(train, test, P, Q, -1, 200);
emb_line=load_embed(sprintf('%s/%s-dataset/data/node_rec/line_embed.txt', data_dir_all, dataset));
P = emb_line;
Q = emb_line;
result_line = evaluate_item(train, test, P, Q, -1, 200);
result = [result_line,result_deepwalk, result_netmf,result_agh_1, result_agh_2,result_imh,result_sh,result_itq,result_svd,result_itq_svd];
ndcg = cell2mat({result.ndcg}');
ndcg = ndcg(:,50);
auc = cell2mat({result.auc}');
mpr = cell2mat({result.mpr}');

final = full([ndcg, auc, mpr]);

dlmwrite(sprintf('%s/%s-dataset/data/node_rec/nr_result_%s.txt', data_dir_all, dataset, dataset), final,'precision','%.4f');

%% testing multi-class classification by varying ratio of subspace learning
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
Ts = [10,10,1,1];
i=4;
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
B_svd = (B_svd>0)*2-1;
num = 10;
timing = zeros(num,1);
data_dir = sprintf('%s/%s-dataset/data/mc_ratio', data_dir_all, dataset);
for i=1:num
    tic;B_itq_svd = dne_ao_itq_subspace(net, B_svd, 'ratio',i*0.1);timing(i)=toc;
    B_itq_svd = (B_itq_svd>0)*2-1;
    file_name = sprintf('%s/embedding_%s_%d_fixiter.txt', data_dir, 'bitq_svd', i);
    fileid = fopen(file_name, 'w');
    fprintf(fileid, '%d %d\n', n, dim);
    fclose(fileid);
    dlmwrite(file_name, [(0:n-1)', B_itq_svd], 'delimiter', ' ', '-append')
end
dlmwrite(sprintf('%s/timing.txt', data_dir), timing);


%% testing node recommendation by varying ration of subspace learning
wdata_dir_all = '/home/dlian/Data/data/network/';
dataset = datasets{i};
load(sprintf('%s/%s-dataset/data/node_rec/%s.mat', data_dir_all, dataset, dataset))
network=train;
T = Ts(i);
b = 1;
dim = 128;
n = length(network);

if T==1
    net = transform_network(network, T, b);
    [U, S] = svds(net, dim);
    V = diag(1./diag(S)) * U.' * net;
else
    load(sprintf('%s/%s-dataset/data/node_rec/netmf.mat', data_dir_all, dataset))
    S = embedding.'*embedding;
    U = embedding*diag(1./sqrt(diag(S)));
    V = diag(1./diag(S)) * U.' * net;
end


B_svd = proj_hamming_balance(U * S);
B_svd = (B_svd>0)*2-1;

num = 10;
timing = zeros(num,1);
%data_dir = sprintf('%s/%s-dataset/data/mc_ratio', data_dir_all, dataset);
result=cell(num,1);
for i=1:num
    tic;B_itq_svd = dne_ao_itq_subspace(net, B_svd, 'ratio',i*0.1);timing(i)=toc;
    B_itq_svd = (B_itq_svd>0)*2-1;
    P=B_itq_svd;
    Q=B_itq_svd;
    result{i} = evaluate_item(train, test, P, Q, -1, 200);
end
result = cell2mat(result);
ndcg = cell2mat({result.ndcg}');
ndcg = ndcg(:,50);
auc = cell2mat({result.auc}');
mpr = cell2mat({result.mpr}');

final = [full([ndcg, auc, mpr]),timing];
dlmwrite(sprintf('%s/%s-dataset/data/nr_ratio.txt', data_dir_all, dataset), final, 'precision','%.4f');

%% multiple running of node recommendation (with 4 more)

%% knn experiments
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
i=3;
dataset = datasets{i};
data_dir_all = '/home/dlian/Data/data/network/';
load(sprintf('%s/%s-dataset/data/node_rec/hash_code.mat', data_dir_all, dataset))
time = zeros(5,2);
for k=1:5
    time(k,1)=knn_search_exp(W_itq_svd,W_itq_svd,k*200);
    time(k,1)=knn_search_exp(W_itq_svd,W_itq_svd,k*200);
end
