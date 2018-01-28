function s=subspace_selection(A, b, d)
%%% min s'*A*s - 2 b'*x, s.t. s in {0,1}
k = size(A,1);
assert(k>d)
s = false(k,1);
[~,ind] = min(diag(A) - 2*b);
s(ind)=true;
for i=2:d
    grad = 2*A*s - 2*b;
    grad(s>0.5) = inf;
    [~,ind] = min(grad);
    s(ind)=true;
end
end