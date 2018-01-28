function mat = load_embed(filename)
head = dlmread(filename,' ', [0, 0, 0, 1]);
n = head(1);
dim = head(2);
mat = zeros(n,dim);
data = dlmread(filename,' ', 1, 0);
ind = data(:,1)+1;
mat(ind,:) = data(:,2:end);
end