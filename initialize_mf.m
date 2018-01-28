function [P, Q] = initialize_mf(net, dim)
    [m, ~] = size(net);
    P = randn(m, dim) * 0.1;
    prev_loss = inf;
    for iter = 1:100
        Q = (pinv(P) * net).';
        P = (pinv(Q) * net.').';
        loss_val = loss_mf(net, P, Q);
        fprintf('initialize_mf with %d th iteration, loss %.3f\n', iter, loss_val)
        if abs(prev_loss - loss_val) < prev_loss * 1e-3
            break
        end
        prev_loss = loss_val;
    end
end