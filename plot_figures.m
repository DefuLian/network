%%
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Wiki';
file_name = sprintf('%s/%s-dataset/data/mc_ratio/mc_ratio.txt', data_dir_all, dataset);
micro = dlmread(file_name, '\t', [1,0,10,4]);
micro = micro([2:end,1],:);
macro = dlmread(file_name, '\t', [12,0,21,4]);
macro = macro([2:end,1],:);
time = dlmread(sprintf('%s/%s-dataset/data/mc_ratio/timing2.txt', data_dir_all, dataset));
time = time(10)./time(1:10);
if strcmp(dataset, 'Wiki')
[time,ind]=sort(time);
micro=micro(ind,:);
macro=macro(ind,:);
end
%time = time(1:9);
%micro=(micro(1:10,:)-repmat(micro(10,:),10,1))./repmat(micro(end,:),10,1) * 100;
%macro=(macro(1:10,:)-repmat(macro(10,:),10,1))./repmat(macro(end,:),10,1) * 100;
%micro = micro(1:9,:);
%macro = macro(1:9,:);
figure('visible','off')
m = {'-o','--+',':x','-.','-^'};
for i=1:size(micro,2)
    plot(time.', micro(:,i), m{i}, 'linewidth',2);hold on
end
%xlim([0.95,1.32]);
%xlim([0.5,1]);%Flickr
%xlim([0.53,1]);%BlogCatalog
%xlim([0.55,1]);%PPI
%ylim([-20,5]);
%set(gca,'ytick',0.85:0.05:1.05)
legend('10%','30%','50%','70%','90%','location','southeast')
xlabel('speedup of training'); ylabel('Micro-F1(%)')
ApplyFigTemplate(gcf,gca);
print(sprintf('%s/%s-dataset/data/mc_ratio/micro_%s.pdf', data_dir_all, dataset, dataset), '-dpdf');
close(gcf);
figure('visible','off')
for i=1:size(macro,2)
    plot(time.', macro(:,i), m{i},'linewidth',2);hold on
end
%xlim([0.95,1.32]);
%xlim([0.5,1]);%Flickr
%xlim([0.53,1]);%BlogCatalog
%xlim([0.55,1]);%PPI
%ylim([0.6,1.05]);
%set(gca,'ytick',0.85:0.05:1.05)
%ylim([-20,5]);
legend('10%','30%','50%','70%','90%','location','southeast')
xlabel('speedup of training'); ylabel('Macro-F1(%)')
ApplyFigTemplate(gcf,gca);
print(sprintf('%s/%s-dataset/data/mc_ratio/macro_%s.pdf', data_dir_all, dataset, dataset), '-dpdf');
close(gcf)

%% plot for node recommendation
datas = cell(4,1);
times = cell(4,1);
data_dir_all = '/home/dlian/Data/data/network/';
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
for i=1:4
dataset = datasets{i};
time = dlmread(sprintf('%s/%s-dataset/data/mc_ratio/timing2.txt', data_dir_all, dataset));
times{i} = time(10)./time(1:10);
file_name = sprintf('%s/%s-dataset/data/nr_ratio.txt', data_dir_all, dataset);
datas{i} = dlmread(file_name);
end
figure('visible','off')
m = {'-o','--+',':x','-.','-^'};
for i= 1:4
    plot(times{i}, datas{i}(:,1), m{i}, 'linewidth',2); hold on;
end
legend(datasets{:}, 'location','southeast');
xlim([1,1.9])
ylim([0.035,0.16])
set(gca, 'ytick', 0.04:0.03:0.16);
ylabel('NDCG@50'); xlabel('speedup of training');

ApplyFigTemplate(gcf,gca);
print(sprintf('%s/nr_ndcg.pdf', data_dir_all), '-dpdf');
close(gcf);

figure('visible','off')
m = {'-o','--+',':x','-.','-^'};
for i= 1:4
    plot(times{i}, datas{i}(:,2), m{i}, 'linewidth',2); hold on;
end
legend(datasets{:},'location','northeast');
xlim([1,1.9])
ylim([0.6,0.9]);
set(gca, 'ytick', 0.6:0.1:0.9);
ylabel('AUC'); xlabel('speedup of training');

ApplyFigTemplate(gcf,gca);
print(sprintf('%s/nr_auc.pdf', data_dir_all), '-dpdf');
close(gcf);

figure('visible','off')
m = {'-o','--+',':x','-.','-^'};
for i= 1:4
    plot(times{i}, datas{i}(:,3), m{i}, 'linewidth',2); hold on;
end
legend(datasets{:},'location','southeast');
xlim([1,1.9])
ylim([0.1,0.4]);
set(gca, 'ytick', 0.1:0.1:0.4);
ylabel('MPR'); xlabel('speedup of training');

ApplyFigTemplate(gcf,gca);
print(sprintf('%s/nr_mpr.pdf', data_dir_all), '-dpdf');
close(gcf);

%%
figure('visible','off')
color = [0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];
figure('visible','off')
data = dlmread(sprintf('%s/knn_ranking_comparison.txt', data_dir_all));
speedup = data(:,2)./data(:,1);
bar(speedup, 'FaceColor',color(1,:),'EdgeColor',color(1,:))
set(gca, 'xticklabel', {'BlogCatalog','PPI', 'Wiki','Flickr'})
%ylim([0,0.3])
xlim([.5,4.5])
ylabel('speedup of hamming ranking');
ApplyFigTemplate(gcf, gca);
print(sprintf('%s/hamming_ranking_speedup.pdf', data_dir_all), '-dpdf');
close(gcf);


%%
figure('visible','off')
m = {'-o','--+',':x','-.','-^'};
datas = zeros(11,4);
data_dir_all = '/home/dlian/Data/data/network/';
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
for i=1:4
dataset=datasets{i};
file_name = sprintf('%s/%s-dataset/data/decorrelation/final.result', data_dir_all, dataset);
micro = dlmread(file_name, '\t', [1,0,11,4]);
datas(:,i) = micro([3:end,1:2],1);
%macro = dlmread(file_name, '\t', [13,0,23,4]);
%macro = macro([3:end,1:2],:);
end
datas = datas(2:end,:)./repmat(datas(1,:),10,1);
gamma=[0.005*2.^(1:10)];
for i=1:4
semilogx(gamma,datas(:,i),m{i},'linewidth',2);hold on
end
ylim([0.7,1.1])
set(gca,'ytick',0.7:0.1:1.1);
legend(datasets{:},'location','southwest');
xlabel('\gamma')
ylabel('relative Micro-F1')
ApplyFigTemplate(gcf, gca);
print(sprintf('%s/sentivity_gamma_micro.pdf', data_dir_all), '-dpdf');
close(gcf);
%%
figure('visible','off')
m = {'-o','--+',':x','-.','-^'};
datas = zeros(11,4);
data_dir_all = '/home/dlian/Data/data/network/';
datasets={'BlogCatalog','PPI','Wiki','Flickr'};
for i=1:4
dataset=datasets{i};
file_name = sprintf('%s/%s-dataset/data/nr_decorrelation.txt', data_dir_all, dataset);
data = dlmread(file_name, ',');
datas(:,i)=data(:,1);
%macro = dlmread(file_name, '\t', [13,0,23,4]);
%macro = macro([3:end,1:2],:);
end
datas = datas(2:end,:)./repmat(datas(1,:),10,1);
gamma=[0.005*2.^(1:10)];
for i=1:4
semilogx(gamma,datas(:,i),m{i},'linewidth',2);hold on
end
ylim([0.7,1.1])
set(gca,'ytick',0.7:0.1:1.1);
legend(datasets{:},'location','southwest');
xlabel('\gamma')
ylabel('relative NDCG@50')
ApplyFigTemplate(gcf, gca);
print(sprintf('%s/sentivity_gamma_ndcg.pdf', data_dir_all), '-dpdf');
close(gcf);

%%
figure('visible','off')
color = [0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];

data_dir = '~/Data/data/network/Flickr-dataset/data/mc_parameter_sensitivity/';
micro = dlmread(sprintf('%s/final.txt', data_dir),'\t', [1,0,26,5]);
semilogx(micro(1:6,end),micro(1:6,5), '-o', 'linewidth',2); hold on;
semilogx(micro(17:22,end),micro(17:22,5), '--+', 'linewidth',2); hold on


%macro = dlmread(sprintf('%s/final.txt', data_dir),'\t', [29,0,54,5]);
%semilogx(macro(1:6,end),macro(1:6,5), '-o', 'Color',color(1,:), 'linewidth',2); hold on;
%semilogx(macro(17:22,end),macro(17:22,5), '--+','Color',color(2,:), 'linewidth',2);
xlim([0,513]);
legend('INH-MF','NetMF', 'location','southeast');
set(gca, 'xtick',micro(1:6,end))
xlabel('dimension of representation space');
ylabel('Micro-F1');
ylim([20,40]);
%text(25,30,'Micro-F1');
%text(25,10,'Macro-F1');
ApplyFigTemplate(gcf, gca);
print(sprintf('%s/ps_mc_dim.pdf', data_dir),'-dpdf');
close(gcf);
%%
figure('visible','off')
plot(micro(7:16,end),micro(7:16,5), '-o', 'linewidth',2); hold on;
plot(micro(23:26,end),micro(23:26,5), '--+', 'linewidth',2); hold on

%plot(macro(7:16,end),macro(7:16,5), '-o', 'Color',color(1,:), 'linewidth',2); hold on;
%plot(macro(23:26,end),macro(23:26,5), '--+','Color',color(2,:), 'linewidth',2);
legend('INH-MF','NetMF', 'location','southeast');
xlabel('percentage of training data');
ylabel('Micro-F1');
set(gca,'xtick',2:2:10)
set(gca,'xticklabel',{'20%','40%','60%','80%','100%'});
%text(5,30,'Micro-F1');
%text(4,15,'Macro-F1');
ApplyFigTemplate(gcf, gca);
print(sprintf('%s/ps_mc_ratio.pdf', data_dir),'-dpdf');
close(gcf);

%%
figure('visible','off')
data_dir = '~/Data/data/network/Flickr-dataset/data/node_rec/node_rec0/';
netmf_ratio = dlmread(sprintf('%s/nr_ratio_metric_netmf.txt',data_dir));
netmf_dim = dlmread(sprintf('%s/nr_dim_metric_netmf.txt',data_dir));

ratio = dlmread(sprintf('%s/nr_ratio_metric.txt',data_dir));
dim = dlmread(sprintf('%s/nr_dim_metric.txt',data_dir));
semilogx(2.^(4:9), dim(:,1),'-o', 'linewidth',2);hold on;
semilogx(2.^(4:9), netmf_dim(:,1),'--+', 'linewidth',2);hold on;
legend('INH-MF','NetMF', 'location','southeast');
xlabel('dimension of representation space');
ylabel('NDCG@50');
xlim([0,513]);
set(gca, 'xtick',2.^(4:9))
ApplyFigTemplate(gcf, gca);
print(sprintf('%s/ps_nr_dim.pdf', data_dir),'-dpdf');
close(gcf);

figure('visible','off')
plot(1:10,ratio(:,1),'-o', 'linewidth',2); hold on
plot(7:10,netmf_ratio(:,1),'-o', 'linewidth',2);
ylim([0,0.15]);
legend('INH-MF','NetMF', 'location','southeast');
xlabel('percentage of training data');
ylabel('NDCG@50');
set(gca,'xtick',2:2:10)
set(gca,'xticklabel',{'20%','40%','60%','80%','100%'});
ApplyFigTemplate(gcf, gca);
print(sprintf('%s/ps_nr_ratio.pdf', data_dir),'-dpdf');
close(gcf);
%%
m = {'-o','--+',':x','-.','-^'};
data_dir = '~/Data/data/network/Flickr-dataset/data';
figure('visible','off')
time_dim = dlmread('~/Data/data/network/Flickr-dataset/data/timing_dim.txt',',');
for i=1:4
semilogx(2.^(4:9),time_dim(:,i),m{i},'linewidth',2);hold on
end
set(gca, 'xtick',2.^(4:9))
ApplyFigTemplate(gcf, gca);
xlabel('dimension of representation space');
legend('INH-MF','SH','ITQ','AGH-1','location','northwest');
ylabel('training time (second)')
print(sprintf('%s/time_dim.pdf', data_dir),'-dpdf');
close(gcf);

figure('visible','off')
time_ratio = dlmread('~/Data/data/network/Flickr-dataset/data/timing_ratio.txt',',');
for i=1:4
plot(1:10,time_ratio(:,i),m{i},'linewidth',2);hold on
end
ApplyFigTemplate(gcf, gca);
legend('INH-MF','SH','ITQ','AGH-1','location','northwest');
set(gca,'xtick',2:2:10);
set(gca,'xticklabel',{'20%','40%','60%','80%','100%'});
xlabel('percentage of training data');
ylabel('training time (second)')
print(sprintf('%s/time_ratio.pdf', data_dir),'-dpdf');
close(gcf);


%%
lin_info = h5read('64/linscan_0_10000_1M.h5','/linscan');
lin_cput_64 = lin_info.cput;
min_info = h5read('64/mih_lsh_0_10000_1M_R0.h5','/mih');
min_cput_64 = min_info.cput;


lin_info = h5read('128/linscan_0_10000_1M.h5','/linscan');
lin_cput_128 = lin_info.cput;
min_info = h5read('128/mih_lsh_0_10000_1M_R0.h5','/mih');
min_cput_128 = min_info.cput;

lin_info = h5read('256/linscan_0_10000_1M.h5','/linscan');
lin_cput_256 = lin_info.cput;
min_info = h5read('256/mih_lsh_0_10000_1M_R0.h5','/mih');
min_cput_256 = min_info.cput;

color = [0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];
m = {'-o','--+',':x','-.','-^'};
n=[10000,50000,100000,300000,500000,700000,900000];
plot(n,lin_cput_64*10000,m{1},'color',color(1,:)); hold on; 
plot(n,lin_cput_128*10000,m{1},'color',color(2,:)); hold on; 
plot(n,lin_cput_256*10000,m{1},'color',color(3,:)); hold on; 
%plot(n,lin_cput_64,m{3},'color',color(1)); hold on; 

plot(n,min_cput_64*10000,m{2},'color',color(1,:)); hold on; 
plot(n,min_cput_128*10000,m{2},'color',color(2,:)); hold on; 
plot(n,min_cput_256*10000,m{2},'color',color(3,:)); hold on; 



%%
dataset = 'Flickr';
data_dir_all = '/home/dlian/Data/data/network/';
rand_loss=load(sprintf('%s/%s-dataset/data/loss_convergence_random.txt', data_dir_all, dataset));
greedy_loss=load(sprintf('%s/%s-dataset/data/loss_convergence_greedy.txt', data_dir_all, dataset));

for i= 1:3
    gl = greedy_loss(i,:);
    gl = gl(gl>0);
    x_gl = 1:length(gl);
    plot(x_gl, gl(x_gl)); hold on;
    rl = rand_loss(i,:);
    rl = rl(rl>0);
    x_rl = 1:length(rl);
    plot(x_rl, rl(x_rl)); hold on;
end