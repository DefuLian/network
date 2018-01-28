function [B, Q] = dne_ao(net, B)
M = net;
Mt = net.';
max_iter = 100;
prev_loss = inf;
n=size(net,1);
k=size(B,2);
for iter=1:max_iter
    Q =  (Mt * B) / (B.' * B + 10*ones(k,1)*ones(1,k));
    %Q = Mt * pinv(B).';
    curr_loss = loss_mf(net, B, Q);
    fprintf('%d iteration, loss %.3f\n', iter-1, curr_loss);
    if abs(prev_loss - curr_loss) < prev_loss * 1e-3
        break
    end
    prev_loss = curr_loss;
    if true
        QtQ = Q.' * Q;
        MQ = M * Q;
        MQt = MQ.';
        parfor i=1:n
            B(i,:) = bqp(QtQ, MQt(:,i));
        end
    else
        hessian = Q.' * Q;
        L = eigs(hessian, 1);
        mq = M * Q;
        for sub_iter = 1:50
            B_0 = B;
            gradient = B * hessian - mq;
            B = prox(B, gradient, 1/L);
            %fprintf('  sub iter %d: norm diff:%.3f\n', sub_iter, norm(B - B_0));
            if norm(B - B_0, 'fro') < eps
                break
            end
        end
    end
end
%fprintf('%d iteration, loss %.3f\n', iter, loss_mf(net, B, Q));
end

function B = prox(B, gradient, lr_0)
    lr = lr_0 *0.8;
    B = proj_hamming_balance(B - lr * gradient);
end





