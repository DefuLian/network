function [P,Q]=randomization(net,dim)
[~,n]= size(net);
Y=net*randn(n,dim*2);
[P,~]=qr(Y,0);
A=P.'*net;
[U,S,V]=svds(A,dim);
P=P*U*S;
Q=V;
end