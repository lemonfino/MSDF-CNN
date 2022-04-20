% matrix:6587*5295
interaction = load('../data/AdjmatrixyizhiDPI.txt');       %%相互作用

% load embedding features
drug_feat = load('../feature/drug_feature100.txt');
prot_feat = load('../feature/protein_feature400.txt');

%十折交叉验证的数据集划分
nFold = 10;        
shape_train=[];    %%训练集大小 
shape_test=[];     %%测试集大小
seed = 2;          %%种子数
rng(seed);
Pint = find(interaction);                                %%已知关联所处的位置索引     正样本
Nint = length(Pint);                                     %%正样本数目
Pnoint = find(~interaction);                             %%未知关联所处的位置索引     总负样本
Pnoint = Pnoint(randperm(length(Pnoint), Nint * 1));     %%随机抽取一定正样本数量的负样本
Nnoint = length(Pnoint);                                 %%负样本数目

posFilt = crossvalind('Kfold', Nint, nFold);             %%正样本经十折交叉验证划分为十份的标签
negFilt = crossvalind('Kfold', Nnoint, nFold);           %%负样本经十折交叉验证划分为十份的标签

	for i = 1 : nFold
        abc=i-1;     
		train_posIdx = Pint(posFilt ~= i);              %%训练集中正样本的位置索引
		train_negIdx = Pnoint(negFilt ~= i);            %%训练集中负样本的位置索引
		train_idx = [train_posIdx; train_negIdx];       %%训练集的位置索引
		Ytrain = [ones(length(train_posIdx), 1); zeros(length(train_negIdx), 1)];    %%训练集标签    正为1，负为0
        shape_train=[shape_train,size(Ytrain,1)];       %%k次中，每次的训练集大小

		test_posIdx =Pint(posFilt == i);                %%测试集中正样本的位置索引
		test_negIdx = Pnoint(negFilt == i);             %%测试集中负样本的位置索引
		test_idx = [test_posIdx; test_negIdx];          %%测试集的位置索引
		Ytest = [ones(length(test_posIdx), 1); zeros(length(test_negIdx), 1)];        %%测试集标签    正为1，负为0
        shape_test=[shape_test,size(Ytest,1)];          %%k次中，每次的测试集大小
        
		[I, J] = ind2sub(size(interaction), train_idx);     %%训练集的行（药物），列（蛋白）  将线性索引转换为下标
        B=drug_feat(I,:);
        C=prot_feat(J,:);
        trainint=[B,C];               %%水平连接   训练集数据的特征
        [k,z] = ind2sub(size(interaction), test_idx);       %%测试集的行（药物），列（蛋白）  将线性索引转换为下标
        D = drug_feat(k,:);
        F = prot_feat(z,:);
        testint=[D,F];                %%水平连接   测试集数据的特征
        
        dlmwrite(['../CNN_Input/Train/train', num2str(abc), '.txt'],trainint, '\t');            %%特征
        dlmwrite(['../CNN_Input/Test/test', num2str(abc), '.txt'],testint, '\t');
        dlmwrite(['../CNN_Input/Trainlabel/trainlabel', num2str(abc), '.txt'],Ytrain, '\t');    %%标签
        dlmwrite(['../CNN_Input/Testlabel/testlabel', num2str(abc), '.txt'],Ytest, '\t');

    end
    
dlmwrite(['../CNN_Input/trainrow.txt'],shape_train, '\t');       %%数据集大小
dlmwrite(['../CNN_Input/testrow.txt'],shape_test, '\t');
       

