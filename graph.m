
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'BlogCatalog';
data_dir = sprintf('%s/%s-dataset/data/', data_dir_all, dataset);
load(sprintf('%s/%s-dataset/data/%s.mat', data_dir_all, dataset, dataset))
itq = load_embed(sprintf('%s/embedding_%s_%d_%d.txt', data_dir, 'bitq_svd', 10,1));
line = load_embed(sprintf('%s/line_embed_1.txt', data_dir));
netmf = load_embed(sprintf('%s/embedding_%s_%d_%d_nn.txt', data_dir, 'svd', 10,1));
mapped_itq = tsne(itq, []);
mapped_netmf = tsne(netmf, []);
mapped_line = tsne(line, []);
save(sprintf('%s/%s-dataset/mapped.mat', data_dir_all, dataset), 'mapped_itq', 'mapped_line', 'mapped_netmf');
%%
[B,ind]= sort(sum(group),'descend');
ind = ind(4:6);
group_filter = group(:,ind);
row_ind = sum(group_filter,2)>0;
label = num2cell(group_filter(row_ind,:),2);
label=cellfun(@find, label, 'UniformOutput',false);
label = cellfun(@(x) x(1), label);
gscatter(mapped_netmf(row_ind,1), mapped_netmf(row_ind,2), label);figure
gscatter(mapped_itq(row_ind,1), mapped_itq(row_ind,2), label);figure
gscatter(mapped_line(row_ind,1), mapped_line(row_ind,2), label);