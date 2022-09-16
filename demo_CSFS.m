clear all;
smalle = 10.^(-5);
d1=32;d2=32;

 load YaleB_32x32; addpath('YaleB_10Train'); data=fea; label=gnd30 ; Initn=38; tr_n=10; l_n=3;
fprintf('Extend Yale B\n');

[nSmp,nFea]=size(fea);
for i=1:nSmp
    data(i,:)=data(i,:)./max(1e-12,norm(data(i,:)));
end
%%  Settings
%flag:   1:CSLS;2:DCSLS
flag=2;
if flag==1
    fprintf('CSLS\n');
else if flag==2
        fprintf('DCSLS\n');
    end
end

cost_setting=[];
cost_setting.C_OI=20;
cost_setting.C_IO=2;
cost_setting.C_II=1;
options.cost_setting=cost_setting;
options.Regu=0.1;

trtime=[];tetime=[];
Idxtrl=[];Idxtru=[];
for i=1:10
    load(int2str(i+9));
    
    Idxtr(i,:)=trainIdx';
    Idxte(i,:)=testIdx';
    
    Idxl=[];Idxu=[];temp=[];
    for j=1:Initn
        Idxl=[Idxl,trainIdx(1+tr_n*(j-1):l_n+tr_n*(j-1),1)'];
        Idxu=[Idxu,trainIdx(l_n+1+tr_n*(j-1):tr_n*j,1)'];
    end
    Idxtrl(i,:)=Idxl;
    Idxtru(i,:)=Idxu;
end
N_test=size(Idxte,2);


rocy=[];
roclabel=[];
for m=2:1:10
    lamda=0.5*(m-1);
    fprintf('******************       lamda=%d            ****************************\n',lamda);
    for i=1:10
        fprintf('Iter %d ',i);
        tr=[];tr_l=[];tr_u=[];te=[];
        tr=data(Idxtr(i,:),:);
        tr_l=data(Idxtrl(i,:),:);
        tr_u=data(Idxtru(i,:),:);
        te=data(Idxte(i,:),:);
        
        train_num=size(tr,1);
        trainl_num=size(tr_l,1);
        trainu_num=size(tr_u,1);
        test_num=size(te,1);
        
        %%  X  view   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        X_train=[];X_trainl=[];X_trainu=[];X_test=[];
        X_train = tr';
        X_mean = mean(X_train,2);
        X_train = X_train - repmat(X_mean,1,train_num);
        X_trainl=tr_l'-repmat(X_mean,1,trainl_num);
        X_trainu=tr_u'-repmat(X_mean,1,trainu_num);
        
        X_test = te';
        X_test = X_test - repmat(X_mean,1,test_num);

        
        
        

        %%                               Comparing methods
        tic;time1=clock;
        if flag==1 %CSLS
            options=[];
            options.Metric = 'Euclidean';
            options.NeighborMode = 'KNN';
            options.k =10;
            options.WeightMode = 'HeatKernel';
            dist=EuDist2(X_trainl');options.t = mean(mean(dist));
            %options.t = 1;
            W = constructW(X_trainl',options);
            lamd=10^(4-m);
            [Fsrank] = CSLS(X_trainl',label(Idxtrl(i,:)),W,cost_setting,lamd);
            X_trainl=X_trainl';X_test=X_test';
            train=X_trainl(:,Fsrank);
            test=X_test(:,Fsrank);
        else if flag==2%DCSLS
                options=[];
                options.Metric = 'Euclidean';
                options.NeighborMode = 'Supervised';
                options.k =0;
                options.WeightMode = 'HeatKernel';
                dist=EuDist2(X_trainl');options.t = mean(mean(dist));
                %options.t = 1;
                options.gnd = label(Idxtrl(i,:));
                W = constructW(X_trainl',options);
                
                options=[];
                options.Metric = 'Euclidean';
                options.NeighborMode = 'Supervised';
                options.k = 0;
                options.WeightMode = 'HeatKernel';
                options.t = mean(mean(dist));
                %options.t = 1;
                options.gnd = label(Idxtrl(i,:));
                Wr = constructWr(X_trainl',3,options);
                
                [Fsrank] = DCSLS(X_trainl',label(Idxtrl(i,:)),W,Wr,cost_setting,lamda);
                X_trainl=X_trainl';X_test=X_test';
                train=X_trainl(:,Fsrank);
                test=X_test(:,Fsrank);
            end
        end
        
        %%       Projection
        MaxDim=size(train,2);
        time2=clock;
        trtime(i)=etime(time2,time1);
        r=0;
        for j=50:50:MaxDim
            tic;time1=clock;
            r=r+1;
            class1=knnclassify(test(:,1:j),train(:,1:j),label(Idxtrl(i,:)),3);
            t1=checkresult(class1,label(Idxte(i,:)),cost_setting);
            result1(i,r)=t1.total_cost;
            err1(i,r)=t1.total_err;
            errig1(i,r)=t1.err_OI;
            errgi1(i,r)=t1.err_IO;
            errgg1(i,r)=t1.err_II;
            time2=clock;tetime(i,r)=etime(time2,time1);
        end
        
    end
    %%  Caluate the results
    for t=1:size(result1,2)
        if length(find(result1(:,t))~=0)<10
            break;
        end
    end
    [pr1_cca,index1 ]= min(mean(result1(:,1:t-1)));
    fprintf('\n\n mean1=%d,ID1=%d,',pr1_cca,index1);fprintf('std1=%d\n',std(result1(:,index1),1));
    fprintf('errig1=%d, std=%d\n',mean(errig1(:,index1))/N_test*100,std(errig1(:,index1),1));
    fprintf('errgi1=%d, estd=%d\n',mean(errgi1(:,index1))/N_test*100,std(errgi1(:,index1),1));
    fprintf('errgg1=%d, std=%d\n',mean(errgg1(:,index1))/N_test*100,std(errgg1(:,index1),1));
    fprintf('err1=%d, std=%d\n',mean(err1(:,index1))/N_test*100,std(err1(:,index1),1));
    
    %     [pr2_cca,index2] = min(mean(result2(:,1:t-1)));
    %     fprintf('\n\n mean2=%d,ID2=%d,',pr2_cca,index2); fprintf('std2=%d\n',std(result2(:,index2),1));
    %     fprintf('errig2=%d, estd=%d\n',mean(errig2(:,index2))/N_test*100,std(errig2(:,index2),1));
    %     fprintf('errgi2=%d, std=%d\n',mean(errgi2(:,index2))/N_test*100,std(errgi2(:,index2),1));
    %     fprintf('errgg2=%d, estd=%d\n',mean(errgg2(:,index2))/N_test*100,std(errgg2(:,index2),1));
    %     fprintf('err2=%d, std=%d\n',mean(err2(:,index2))/N_test*100,std(err2(:,index2),1));
    
    fprintf('training time=%d, std=%d,          testing time=%d, std=%d \n',mean(trtime),std(trtime,1),mean(tetime(:,index1)),std(tetime(:,index1),1));
    result1=[];result2=[];err1=[];errig1=[];errgi1=[];errgg1=[];err2=[];errig2=[];errgi2=[];errgg2=[];
    
   % pr1(m,n)=pr1_cca;
    %     pr2(m,n)=pr2_cca;
end


