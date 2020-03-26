function P = trr_test(X, M)
% 2019-01-05
% tensor ridge regression
% X: L x W x H x... I; M{i}: i=1, 2, 3...
X = tensor(X);
[nM, nR, nC] = size(M);
if nC > 1
    P = zeros(nC, X.size(end));
    for c = 1 : nC
        for i = 1 : nR
            t = double(squeeze(ttm(X, M(:,i,c), 1:nM)));
            P(c,:) = P(c,:) + t(:)';
        end
    end
else
    P = 0;
    for i = 1 : nR
        P = P + ttm(X, M(:,i,1), 1:nM);
    end
    P = P.data;
    P = squeeze(P);
end
end