tic
cvx_begin
    variable XX(600, 600) symmetric semidefinite
    minimize(trace(L*XX))
    subject to
        diag(XX) == 1;
cvx_end
toc

m = 16; n = 8;
A = randn(m,n);
b = randn(m,1);

cvx_begin
    variable x(n);
    minimize( norm(A*x-b) );
cvx_end
