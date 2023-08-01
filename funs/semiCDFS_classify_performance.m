function [result] = semiCDFS_classify_performance(Xl, Yl, Xu, Yu, feature_number_list, index, option)
X = [Xl; Xu];
Y = [Yl; Yu];
c = length(unique(Y));   % class number of data
YL = zeros(size(Xl, 1),c);
for i = 1:size(Xl, 1)
   YL(i, Yl(i)) = 1;
end
result = zeros(5, length(feature_number_list));
%----------- Feature selection by the proposed method-----------------%
for i_fea = 1:length(feature_number_list)
    clear Xp;
    option.k = feature_number_list(i_fea); 
    t1 = clock;
    [~, f] = SSCDFS(Xl', YL, Xu', option);
    t2 = clock;
    Xp = X(:, f);
    kfold = 5;
    for t = 1:kfold
        [Xp_tr, Xp_te, Y_tr, Y_te] = load_data_classify(Xp, Y, index, t);
        model = ClassificationKNN.fit(Xp_tr,Y_tr,'NumNeighbors',5);
        predY_te = predict(model, Xp_te);
        res(t,:) = ClusteringMeasure1(Y_te, predY_te);
    end
    result(1:4, i_fea) = mean(res,1)';
end
end