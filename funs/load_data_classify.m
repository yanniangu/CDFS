function [trainset, testset, trainlabel, testlabel] = load_data_classify(data, label, index, t)
% k-fold crossValidation, divive data into k groups according to index
% One groups for testing and the other for training;
% reture the t-th trainset and testset
if nargin < 4
    t = 5;
end
test = (index == t); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
train = (index ~= t);
trainset=data(train,:); %�����ݼ��л��ֳ�train����������
trainlabel=label(train,:); %����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
testset=data(test,:); %test������
testlabel=label(test,:);
end

