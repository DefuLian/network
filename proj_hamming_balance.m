function B = proj_hamming_balance(X)
    n = size(X, 1);
    c = median(X);
    B = (X - repmat(c, n, 1) >  - eps) * 2 - 1;
end