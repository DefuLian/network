function [B, Q] = dne_ao_itq_subspace(net, B, varargin)
[ratio, gamma] = process_options(varargin, 'ratio',1, 'gamma',0);
M = net;
Mt = net.';
max_iter = 200;
prev_loss = inf;
n = length(net);
k = size(B,2);
mtb = Mt * B;
d = k*ratio;
for iter=1:max_iter
    Q=proj_stiefel_manifold(mtb);
    if gamma>0
        B_h=sqrt(n)*proj_stiefel_manifold(B);
    else
        B_h = zeros(size(B));
    end
    curr_loss = loss_mf(net, B, Q) + gamma/2*sum(sum((B-B_h).^2));
    fprintf('%3d iteration, loss %.3f\n', iter, curr_loss);
    %if abs(prev_loss - curr_loss) < 1e-6
    %    break
    %end
    prev_loss = curr_loss;
    mqb = M*Q+gamma*B_h;
    if d<k
        A = (1+gamma)*(B.')*B; b = sum(B.*mqb); s=subspace_selection(A, b.',d);
        mtb(:,s) = mtb(:,s) - Mt * B(:,s);
        B(:,s) = proj_hamming_balance(mqb(:,s));
        mtb(:,s) = mtb(:,s) + Mt * B(:,s);
    else
        B=proj_hamming_balance(mqb);
        mtb=Mt*B;
    end
    %else
    %    [U1,~,V1]=svd(Mt*B(:,s),0); k1 = sum(s);
    %    [ur,~]=qr(U.'*U1); 
    %    ur=ur(:,(k1+1):end); 
    %    UU=[U1, U*ur];
    %    V1=[V1;zeros(k-k1,k1)];
    %    [vr,~]=qr(V.'*V1);
    %    vr=vr(:,(k1+1):end);
    %    VV=[V1,V*vr];
    %    Q = UU * VV;
    %end
end
end

function val = loss_mf(net, P, Q)
    val = sum(sum(net.^2)) - 2 * sum(sum((P.' * net) .* Q.')) + sum(sum((Q.' * Q) .* (P.' * P)));
    val = val / 2;
end


