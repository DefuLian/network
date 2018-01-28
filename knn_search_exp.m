function t=knn_search_exp(X,Y,K)
if isa(X, 'uint8') && isa(Y,'uint8')
    tic;knnsearch(X, Y, 'Distance', @(ZI, ZJ) hammingDist(ZI, ZJ).', 'K', K);t=toc;
else
    tic;knnsearch(X, Y, 'Distance', @(ZI, ZJ) ZJ * ZI.');t=toc;
end
end