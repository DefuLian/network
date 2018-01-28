function mat = transform_network(network, T, b)
assert(T<5, 'The order of proximity should not be large for efficiency')
n = length(network);
if issymmetric(network)
    vol = sum(sum(network));
    d = sum(network) + 1e-10;
    D = spdiags(1./d.', 0, n, n);
    X = D * network;
    power = X;
    mat = X;
    for t = 1:(T-1)
        mat = mat + power * X;
        power = power * X;
    end
    mat = mat * D;
    mat = mat .* (vol / T / b);
    %mat(mat<=1) = 0;
    [I, J, K] = find(mat);
    ind = K>1;
    mat = sparse(I(ind), J(ind), log(K(ind)), n, n);
else
    d_row = sum(network, 2) + 1e-10;
    D_row = spdiags(1./d_row, 0, n, n);
    A = D_row * network;
    mat = A;
    for t=1:(T-1)
        mat = mat * A;
    end
    d_col = sum(mat) + 1e-10;
    D_col = spdiags(1./d_col.', 0, n, n);
    mat = mat * D_col;
    mat = mat .* (n / b);
    [I, J, K] = find(mat);
    K = log(K);
    ind = K>0;
    mat = sparse(I(ind), J(ind), K(ind), n, n);
end
end