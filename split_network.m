function [train,test]= split_network(network, mode, ratio)
n = length(network);
[J,I,K]=find(network.');
ind=I<J;
asys_network=sparse(I(ind),J(ind),K(ind),n,n);
[asys_train, asys_test]=split_matrix(asys_network, mode, ratio);
train = asys_train + asys_train.';
test = asys_test + asys_test.';
end