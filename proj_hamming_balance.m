function B = proj_hamming_balance(X)
    n = size(X, 1);
    c = median(X);
    B = sign(X - repmat(c, n, 1));
end