function mat = transform_network_large(network, T)
assert(T<5, 'The order of proximity should not be large for efficiency')
n = length(network);
if issymmetric(network)
end
end