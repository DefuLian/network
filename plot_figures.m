%%
data_dir_all = '/home/dlian/Data/data/network/';
dataset = 'Flickr';
file_name = sprintf('%s/%s-dataset/data/mc_ratio/mc_ratio.txt', data_dir_all, dataset);
micro = dlmread(file_name, '\t', [1,0,10,4]);
micro = micro([2:end,1],:);
macro = dlmread(file_name, '\t', [12,0,21,4]);
macro = macro([2:end,1],:);
time = dlmread(sprintf('%s/%s-dataset/data/mc_ratio/timing.txt', data_dir_all, dataset));
time = time(1:end)./time(end);
%micro=micro(1:end,:)./repmat(micro(end,:),10,1);
%macro=macro(1:end,:)./repmat(macro(end,:),10,1);
figure('visible','off')
m = {'-o','--+',':x','-.','-^'};
for i=1:size(micro,2)
    plot(time.', micro(:,i), m{i});hold on
end
xlim([0.4,1]);
%ylim([0.85,1.05]);
%set(gca,'ytick',0.85:0.05:1.05)
legend('10%','30%','50%','70%','90%','location','southeast')
xlabel('speedup of training'); ylabel('Micro(%)')
ApplyFigTemplate(gcf,gca);
print(sprintf('%s/%s-dataset/data/mc_ratio/micro_%s.pdf', data_dir_all, dataset, dataset), '-dpdf');
close(gcf);
figure('visible','off')
for i=1:size(macro,2)
    plot(time.', macro(:,i), m{i});hold on
end
xlim([0.4,1]);
%ylim([0.85,1.05]);
%set(gca,'ytick',0.85:0.05:1.05)
legend('10%','30%','50%','70%','90%','location','southeast')
xlabel('speedup of training'); ylabel('Macro(%)')
ApplyFigTemplate(gcf,gca);
print(sprintf('%s/%s-dataset/data/mc_ratio/macro_%s.pdf', data_dir_all, dataset, dataset), '-dpdf');
close(gcf)
