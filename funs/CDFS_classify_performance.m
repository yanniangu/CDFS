function [ result] = CDFS_classify_performance( X, Y, index, feature_number_list)
%
%   X: N x d; the last column is label
addpath('funs');
n = size(X, 1);
c = length(unique(Y));
YL = zeros(n,c);
for i = 1:n
    YL(i,Y(i)) = 1;
end
result = zeros(5, length(feature_number_list));
for i_fea = 1:length(feature_number_list)
    clear Xp;
    k = feature_number_list(i_fea); 
    t1 = clock;
    [~, f] = CDFS(X, YL, k);
    t2 = clock;
    Xp = X(:, f);
    kfold = 5;
    for t = 1:kfold
        [Xp_tr, Xp_te, Y_tr, Y_te] = load_data_classify(Xp, Y, index, t);
        model = ClassificationKNN.fit(Xp_tr,Y_tr,'NumNeighbors',5);
        predY_te = predict(model, Xp_te);
        res(t,:) = ClusteringMeasure(Y_te, predY_te);
    end
    result(1:4, i_fea) = mean(res,1)';
    result(5, i_fea) = etime(t2, t1);
end
end



