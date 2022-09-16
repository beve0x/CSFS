clear all;
addpath('Acc_sin_5pTrain')%using 5% percent data for model training with the remaining data for testing
addpath('libsvm-3.23/matlab')%SVM fold

load Accident_Singapore.mat;%input the accident data
fprintf('Accident prediction in Singapore\n');
gnd(find(gnd==0),1)=-1;%convert the label into {-1,1}, where -1 denotes non-accident data and "1" denote accident data 
data=data(:,[2:11,13:end]);% accident data

[nSmp,nFea]=size(data);
%pre-proceesing data
for i=1:nSmp
    data(i,:)=data(i,:)./max(1e-12,norm(data(i,:)));
end
%%  Settings
%class imbalanced ratio=74514/6194=12
%I: cost for non-accident samples
%O: cost for accident samples
cost_setting=[];
cost_setting.C_OI=10;
cost_setting.C_IO=1;

cost_setting_t=[];
cost_setting_t.C_OI=10;
cost_setting_t.C_IO=1;

flag=3; %=1: CSFS; =2: CSLS; =3: DCSLS; =4: do not feature selection

%% generate the training and test data
trtime=[];tetime=[];
Idxtrl=[];Idxtru=[];

% get the index for training and test 
for i=1:1
    load(int2str(i+9));
    
    Idxtr(i,:)=trainIdx';
    Idxte(i,:)=testIdx';
end
test_num=size(Idxte,2);
train_num=size(Idxtr,2);


%% main code of accident prediction
for i=1:1
    fprintf('Iter %d ',i);
    tr=[];te=[];
    tr=data(Idxtr(i,:),:);
    te=data(Idxte(i,:),:);
    
    %%  mean normalized to zero %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X_train=[];X_test=[];
    X_train = tr;
    X_mean = mean(X_train,1);
    X_train = X_train - repmat(X_mean,train_num,1);
    
    X_test = te;
    X_test = X_test - repmat(X_mean,test_num,1);
    train=X_train;
    test=X_test;
    
    
    %% training for feature selection
    tic;time1=clock;
    %% cost-sensitive fisher score
    if flag==1
        [Fsrank,Fsvalue] = CSFS(X_train,gnd(Idxtr(i,:)),cost_setting.C_IO,cost_setting.C_OI);
        %Fsrank:the id of features ranked according to their importance
        %Fvalue:the importance of each feature
        train=X_train(:,Fsrank(1:end));%re-orginize the training data data accroding to their Fsrank
        test=X_test(:,Fsrank(1:end));%re-orginize the test data data accroding to their Fsrank
        
        %% cost-sensitive laplacian score
    else if flag==2
            %W:KNN similarity graph
            options=[];
            options.Metric = 'Euclidean';
            options.NeighborMode = 'KNN';
            options.k =10;
            options.WeightMode = 'HeatKernel';
            dist=EuDist2(X_train);options.t = mean(mean(dist));
            %options.t = 1;
            W = constructW(X_train,options);
            
            lamd=10^-5;%parameter of CSLS
            [Fsrank,Fsvalue] = CSLS(X_train,gnd(Idxtr(i,:)),W,cost_setting,lamd);
            train=X_train(:,Fsrank(1:end));
            test=X_test(:,Fsrank(1:end));
            
            %% discriminative cost-sensitive laplacian score
        else if flag==3
                %W and Wr: two KNN graph 
                options=[];
                options.Metric = 'Euclidean';
                options.NeighborMode = 'Supervised';
                options.k =0;
                options.WeightMode = 'HeatKernel';
                dist=EuDist2(X_train);options.t = mean(mean(dist));
                %options.t = 1;
                options.gnd = gnd(Idxtr(i,:));
                W = constructW(X_train,options);
                
                options=[];
                options.Metric = 'Euclidean';
                options.NeighborMode = 'Supervised';
                options.k = 0;
                options.WeightMode = 'HeatKernel';
                options.t = mean(mean(dist));
                %options.t = 1;
                options.gnd = gnd(Idxtr(i,:));
                Wr = constructWr(X_train,3,options);
                
                lamda=0.5;%parameter of DCSLS
                [Fsrank,Fsvalue] = DCSLS(X_train,gnd(Idxtr(i,:)),W,Wr,cost_setting,lamda);
                train=X_train(:,Fsrank);
                test=X_test(:,Fsrank);
            else if flag==4
                end
            end
        end
    end
    
    
    %% testing
    time2=clock;
    trtime(i)=etime(time2,time1);
    MaxDim=size(train,2);
    r=0;
    for j=1:1:MaxDim
        tic;time1=clock;
        r=r+1;
        
        %% knn classifier
        %class=knnclassify(test(:,1:j),train(:,1:j),gnd(Idxtr(i,:)),1);
        
        %% CS-SVM
        DD=EuDist2(train(:,1:j),train(:,1:j));
        g=mean(mean(DD));
        
        %SVM training with linear kernel
        %model = svmtrain(l_train, X_trainl', '-s 0 -t 0 -c 3000 -b 1 -h 0 -w1 60');
        
        %SVM training with gassian kernel
        model = svmtrain(gnd(Idxtr(i,:)), train(:,1:j), '-s 0 -t 2 -c 3000 -b 1 -h 0 -w1 10 -g g');
        
        %SVM classification
        [class, accuracy, prob_estimates] = svmpredict(gnd(Idxte(i,:)),  test(:,1:j), model);
        auc(i,r) = roc_curve(prob_estimates',gnd(Idxte(i,:))')
        
        %% calculate acc; f1 measure
        acc(i,r)=length(find(class-gnd(Idxte(i,:))==0))/test_num;%Accuracy
        t1=checkresult(class,gnd(Idxte(i,:)),cost_setting_t);
        result1(i,r)=t1.total_cost;%total cost
        err1(i,r)=t1.total_err;
        errig1(i,r)=1-t1.err_OI;%1-sensitivity
        errgi1(i,r)=1-t1.err_IO;%1-specificity
        
        time2=clock;tetime(i,r)=etime(time2,time1);
    end
end

ff=0;
for t=1:size(result1,2)
    if length(find(result1(:,t)==0))>0
        ff=1;
        break;
    end
end
if ff==1
    [pr1_cca,index1 ]= min(mean(result1(:,1:t-1),1));
else
    [pr1_cca,index1 ]= min(mean(result1,1));
end
fprintf('\n\n mean1=%d,ID1=%d,',pr1_cca,index1);fprintf('std1=%d\n',std(result1(:,index1),1));
fprintf('sensitivity=%d, std=%d\n',mean(errig1(:,index1))*100,std(errig1(:,index1),1));
fprintf('specificity=%d, estd=%d\n',mean(errgi1(:,index1))*100,std(errgi1(:,index1),1));
fprintf('err1=%d, std=%d\n',mean(err1(:,index1))*100,std(err1(:,index1),1));

% fprintf('\n\n mean1=%d',mean(result1));fprintf('std1=%d\n',std(result1,1));
% fprintf('sensitivity=%d, std=%d\n',100-mean(errig1)*1000,std(errig1,1));
% fprintf('Specificity=%d, estd=%d\n',100-mean(errgi1)*1000,std(errgi1,1));
% fprintf('err1=%d, std=%d\n',mean(err1)*1000,std(err1,1));
