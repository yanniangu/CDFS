addpath('datasets');
addpath('funs');
data_list = {'JAFFE','imm40','FERET','FEI','COIL20','ORL','PIX_32_32','Yale','UMIST','USPS'};

for da = 1: length(dataname)
    % Load Data
    load(dataname{da});
    [n, d] = size(X);
    
    kfold = 5;
    index = crossvalind('Kfold',X(1:n,d),kfold);
    
    % Parameters
    feature_number_list = 5:5:30;
    
    % CDFS
    [ KNN_result_CDFS ] = CDFS_classify_performance(X, Y, index, feature_number_list);

    KNN_result_ACC = [KNN_result_CDFS(1,:)];
    KNN_result_NMI = [KNN_result_CDFS(2,:)];
    KNN_result_Fsc = [KNN_result_CDFS(3,:)];
    KNN_result_ARI = [KNN_result_CDFS(4,:)];
    result_name = strcat('CDFS_result_', num2str(label_rate(lr)));
    save(strcat('.\results\',dataname{da},'_',result_name,'.mat'),'KNN_result_ACC','KNN_result_NMI', 'KNN_result_Fsc', 'KNN_result_ARI');
end
