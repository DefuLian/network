function [P,Q]=mf_itq(net,k)
%%% |R-PQ'|_F^2, s.t. Q'*Q=I
R=net;
Rt=net.';
[~,n]=size(R);
P=randn(n,k)*0.01;
max_iter=1000;
prev_loss = inf;
for iter=1:max_iter
    Q=proj_stiefel_manifold(Rt*P);
    curr_loss = loss_mf(net, P, Q);
    fprintf('%3d iteration, loss %.3f\n', iter, curr_loss);
    if abs(prev_loss - curr_loss) < 1e-1
        break
    end
    prev_loss = curr_loss;
    P = R * Q;
end
end

function val = loss_mf(net, P, Q)
    val = sum(sum(net.^2)) - 2 * sum(sum((P.' * net) .* Q.')) + sum(sum((Q.' * Q) .* (P.' * P)));
    val = val / 2;
end