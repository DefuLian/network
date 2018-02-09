function [B, Q] = dne_ao_itq(net, B, gamma)
[m, d]=size(net);
assert(m>=d);
M = net;
Mt = net.';
max_iter = 50;
%prev_loss = inf;
for iter=1:max_iter
    if m==d
        Q = proj_stiefel_manifold(Mt * B);
    else
        Q = proj_stiefel_manifold(M* (Mt * B));
    end
    B_ = sqrt(m) * proj_stiefel_manifold(B);
    %curr_loss = loss_mf(net, B, Q) + gamma / 2 * sum(sum((B - B_).^2));
    %fprintf('%d iteration, loss %.3f\n', iter-1, curr_loss);
    %if abs(prev_loss - curr_loss) < 1e-6
    %    break
    %end
    %prev_loss = curr_loss;
    if m==d
        B = proj_hamming_balance(M * Q + gamma * B_);
    else
        B = proj_hamming_balance(M * (Mt* Q) + gamma * B_);
    end
end
%fprintf('%d iteration, loss %.3f\n', iter, loss_mf(net, B, Q) + gamma / 2 * sum(sum((B - B_).^2)));
end

function val = loss_mf(net, P, Q)
    M = net;
    Mt = net.';
    [m,d]=size(net);
    if m == d
        val = sum(sum(net.^2)) - 2 * sum(sum((P.' * net) .* Q.')) + sum(sum((Q.' * Q) .* (P.' * P)));
    else
        val = sum(sum((Mt*M).^2)) - 2 * sum(sum(((P.' * M) * Mt) .* Q.')) + sum(sum((Q.' * Q) .* (P.' * P)));
    end
    val = val / 2;
end


