function time = testing_time(network, dim)
time = zeros(4,1);
net = transform_network(network, 1, 1);
tic
[U, S] = svds(net, dim);
B_svd = proj_hamming_balance(U * S);
dne_ao_itq(net, B_svd, 0.);
time(1)=toc;

tic;
SHparam.nbits = dim; % number of bits to code each sample
SHparam = trainSH(network, SHparam);
compressSH(network, SHparam); time(2)=toc;

tic; B_itq = compressITQ(network,dim);  B_itq * 2 - 1; time(3)=toc;


tic
[~,anchor]=litekmeans(network,1000);
anchor = sparse(anchor);
s = 5;
B_AGH_1 = OneLayerAGH_Train(network, anchor, dim, s, 0);
B_AGH_1*2-1;
time(4)=toc;

end