function [trainset, testset, trainlabel, testlabel] = load_data_classify(data, label, index, t)
% k-fold crossValidation, divive data into k groups according to index
% One groups for testing and the other for training;
% reture the t-th trainset and testset
if nargin < 4
    t = 5;
end
test = (index == t); %获得test集元素在数据集中对应的单元编号
train = (index ~= t);
trainset=data(train,:); %从数据集中划分出train样本的数据
trainlabel=label(train,:); %获得样本集的测试目标，在本例中是实际分类情况
testset=data(test,:); %test样本集
testlabel=label(test,:);
end

