addpath('datasets');
addpath('funs');
dataname = {'COIL20', 'FEI', 'FERET', 'imm40', 'JAFFE', 'ORL', 'PIX_32_32', 'UMIST', 'Yale'};
label_rate = 0.1:0.1:0.9;

for lr = 1 : length(label_rate)
    for da = 1: length(dataname)
        % data initialization
        load(dataname{da});
        X = full(real(X));
        [n,d]=size(X);
        c=length(unique(Y));
        left = randperm(n);
        se=sort(left(1:round(n*label_rate(lr))),'ascend');
        Xl=X(se,:);
        Yl=Y(se,:);
        Xu=X; 
        Xu(se,:)=[];
        Yu=Y;
        Yu(se,:)=[];
        
        % KNN settings
        kfold = 5;
        index = crossvalind('Kfold',X(1:n,d),kfold);
 
        % Parameters
        feature_number_list = 5:5:30;
        option = [];
        option.r1 = 2.5;
        option.MaxIter = 50;
        option.rho = 1.1;
        option.mu = 0.02;
        option.c = c;
        option.initW = 0;
        
        % semiCDFS
        [ KNN_result_SemiCDFS ] = semiCDFS_classify_performance(Xl, Yl, Xu, Yu, feature_number_list, index, option);
        
        % Results
        KNN_result_ACC= [KNN_result_SemiCDFS(1,:)];
        KNN_result_NMI= [KNN_result_SemiCDFS(2,:)];
        KNN_result_F= [KNN_result_SemiCDFS(3,:)];
        KNN_result_ARI= [KNN_result_SemiCDFS(4,:)];
        result_name = strcat('semiCDFS_result_', num2str(label_rate(lr)));
        save(strcat('.\results\',dataname{da},'_',result_name,'.mat'),'KNN_result_ACC','KNN_result_NMI', 'KNN_result_F', 'KNN_result_ARI');
    end
end
