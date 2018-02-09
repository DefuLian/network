function t=knn_search_exp(X,Y,K)
if isa(X, 'uint8') && isa(Y,'uint8')
    tic;knnsearch(X, Y, 'Distance', @(ZI, ZJ) han_dist(ZJ,ZI), 'K', K);t=toc;
else
    tic;knnsearch(X, Y, 'Distance', @(ZI, ZJ) mult(ZJ,ZI), 'K', K);t=toc;
end
end