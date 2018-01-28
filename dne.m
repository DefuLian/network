function [B, Q] = dne(net, varargin)

[alpha, T, lambda, b, dim, gamma] = process_options(varargin, 'alpha', 0, 'T', 1, 'lambda', 0, 'neg_sum', 5, 'dim', 128, 'gamma', 0.1);
net = transform_network(net, T, b);
if alpha > 0
    [B, Q] = dne_wao(net, dim, lambda);
else
    [B, Q] = dne_ao(net, dim, lambda, gamma);
end
end


