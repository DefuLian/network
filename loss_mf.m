function val = loss_mf(net, P, Q)
    val = sum(sum(net.^2)) - 2 * sum(sum((P.' * net) .* Q.')) + sum(sum((Q.' * Q) .* (P.' * P)));
    val = val / 2;
end