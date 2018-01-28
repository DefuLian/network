function [B, Q] = dne_wao(R, varargin)
%%% |(W.^0.5).*(R - B*Q')|_F^2, s.t. Q'Q=I, B'1=0
%%% When B fixed, min_{Q}1/2|(W.^0.5).*(R - B*Q')|_F^2, s.t. Q'Q=I, which use ADMM
%%% for optimization.
%%%     Q(t+1)=argmin_{Q}1/2|(W.^0.5).*(R - B*Q')|_F^2 + rho/2*|Q-(O(t)-D(t))|_F^2
%%%     O(t+1)=argmin_{O}1/2|Q(t+1)+D(t)-O|_F^2, s.t. O'*O =I
%%%     D(t+1)=D(t)+(Q(t+1)-O(t+1))
%%% 1/2|(W.^0.5).*(R - B*Q')|_F^2 + rho*<D,Q-O> + rho/2*|Q-O|_F^2
%%% When Q fixed, min_{B}|(W.^0.5).*(R - B*Q')|_F^2 + gamma * |B'1|^2
%%% 
%%%
[alpha,k,max_iter,B,rho,gamma]=process_options(varargin, 'alpha',4, 'k',128, 'max_iter',10, 'B',[],...
    'rho',10,'gamma',0.01);
W = (R~=0)*alpha;
[B1,Q]=initialize(R,W,k);
if isempty(B)
    B=proj_stiefel_manifold(B1);
end
prev_loss = fast_loss(R, W, B, Q); %+ gamma* sum(sum(B).^2);
for iter=1:max_iter
    Q = learn_Q_orthogonal2(R,W,B,Q,rho);
    B = learn_B(R,W,B,Q,gamma);
    curr_loss=fast_loss(R, W, B, Q); %+ gamma* sum(sum(B).^2);
    if abs(curr_loss-prev_loss) < 1e-3*prev_loss
        break;
    end
    curr_loss = fast_loss(R, W, B, Q);
    fprintf('dne_wao:%d iter, loss %.3f\n', iter, curr_loss);
    prev_loss = curr_loss;
end

end

function B = learn_B(R, W, B, Q, gamma, binary)
if nargin < 6
    binary = true;
end
Rt = R.';
Wt = W.';
m = size(R,1);
Qt = Q.';
QtQ = Qt * Q;
k = size(B,2);
%b_ = sum(B);
parfor i=1:m
    w = Wt(:,i);
    r = Rt(:,i);
    ind = w > 0;
    sub_Q = Q(ind,:);
    if(nnz(ind) == 0)
        Wi = zeros(0);
    else
        Wi = diag(w(ind));
    end
    A = QtQ+sub_Q.'*Wi*sub_Q;
    y = Qt*(w.*r+r);
    if binary
        %b = bqp(A, y-gamma/2*b_.');
        B(i,:) = bqp((A+A.')/2,y);
    else
        B(i,:) = (A+gamma*eye(k))\y;
    end
    %b_ = b_ - B(i,:) + b.';
    %B(i,:) = b;
end
end

function Q = learn_Q(R, W, B, Q, rho)
[m,n] = size(R);
k = size(B, 2);
Q = optimize(R, W, [B,zeros(m,1)], [Q,zeros(n,1)], zeros(n,k+1), rho, ones(n,1), ones(m,1), zeros(m,1),'ALS');
Q = Q(:,1:k);
end
function Q = learn_Q_orthogonal(R, W, B, Q, rho, max_iter)
[m,n] = size(R);
k = size(B, 2);
D = zeros(n, k);
O = proj_stiefel_manifold(D+Q);
prev_loss = fast_loss(R, W, B, Q)+rho*sum(dot(D,Q-O))+rho/2*sum(dot(Q-O,Q-O));
for iter=1:max_iter
    Q = optimize(R, W, [B,zeros(m,1)], [Q,zeros(n,1)], [O-D,zeros(m,1)], ...
        rho, ones(n,1), ones(m,1), zeros(m,1),'ALS');
    Q = Q(:,1:k);
    O = proj_stiefel_manifold(D+Q);
    D = D+Q-O;
    curr_loss = fast_loss(R, W, B, Q)+rho*sum(dot(D,Q-O))+rho/2*sum(dot(Q-O,Q-O));
    fprintf('  q-learn:%d iteration, loss:%.3f, residual:%.3f\n', iter, curr_loss, norm(Q-O,'fro'));
    if curr_loss>prev_loss || abs(prev_loss-curr_loss)<prev_loss*1e-3
        break
    end
    prev_loss = curr_loss;
end
end

function Q = learn_Q_orthogonal2(R, W, B, Q, rho)
[m,n] = size(R);
k = size(B, 2);
O = proj_stiefel_manifold(Q);
Q = optimize(R, W, [B,zeros(m,1)], [Q,zeros(n,1)], [O,zeros(m,1)], ...
        rho, ones(n,1), ones(m,1), zeros(m,1),'ALS');
Q = Q(:,1:k);
end



function P = optimize(Rt, Wt, Q, P, XU, reg_u, a, d, bI, method)
XU = reg_u * XU; % M x K
XUt = XU.';
K = size(Q,2);
[~, M] = size(Wt);
Qt = Q.';
QtQ = Qt * spdiags(d, 0, length(d), length(d)) * Q;
bR = Qt * (bI .* d);
dR = Qt * spdiags(d, 0, length(d), length(d)) * Rt;
parfor i = 1 : M
    w = Wt(:, i);
    r = Rt(:, i);
    au = a(i);
    if strcmp(method, 'CD')
        P(i,:) = piccf_sub(+r, w, Q, QtQ, P(i,:), XUt(:,i), au, bR, dR(:,i), reg_u, bI);
    elseif strcmp(method, 'ALS')
        Ind = w>0;     
        if(nnz(Ind) == 0)
            Wi = zeros(0);
        else
            Wi = diag(w(Ind));
        end
        sub_Q = Q(Ind,:);
        QCQ = sub_Q.' * Wi * sub_Q + au * QtQ + reg_u * eye(K); %Vt_minus_V = sub_V.' * (Wi .* sub_V) + invariant;
        %Y = Qt * (w .* r - w .* bI + au * (d .* r)) - au * bR + XUt(:,i) ;
        Y = Qt * (w .* r - w .* bI ) + au * dR(:,i) - au * bR + XUt(:,i) ;
        P(i,:) = QCQ \ Y;
    else
        error('Unsupported optimization method')
    end
end
end

function [B, Q]=initialize(R, W, k)
[m,n]=size(R);
randn('state', 10);
B = randn(m,k)*0.01;
Q = randn(n,k)*0.01;
%[B,Q]=piccf(R, 'K',k,'max_iter',10,'alpha',5,'P',B,'Q',Q,'method','ALS');
max_iter = 10;
prev_loss=fast_loss(R,W,B,Q);
for iter=1:max_iter
    B = learn_B(R,W,B,Q,0.01,false);
    Q = learn_Q(R,W,B,Q,0.01);
    curr_loss=fast_loss(R,W,B,Q);
    fprintf('  init:%d iter, loss %.3f\n', iter, curr_loss);
    prev_loss=curr_loss;
end
end
