clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Iris Flower Classification
%  Algorithm       : Voting Ensemble – from scratch
%  
%  Description     : Manual implementation of Voting Ensemble classifier 
%                    on the Iris dataset.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/ensemble-learning/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background and video explanations, 
%  please visit OnlineBME.
%% ========================================================================
load('fisheriris.mat')
%% step 1: devide data into train and test
% using k-fold cross validation method
data1= iris(:,label==1);
data2= iris(:,label==2);
data3= iris(:,label==3);
K=5;
fold= floor(size(data1,2) / K);
Ct=0;
for iter=1:K
    indtest= (iter-1)*fold+1 : iter*fold;
    indtrain= 1:size(data1,2);
    indtrain(indtest)=[];
    
    datatrain1= data1(:,indtrain);
    datatest1= data1(:,indtest);
    
    datatrain2= data2(:,indtrain);
    datatest2= data2(:,indtest);
    
    datatrain3= data3(:,indtrain);
    datatest3= data3(:,indtest);
    
    datatrain= [datatrain1,datatrain2,datatrain3];
    dtrain= [ones(1,size(datatrain1,2)),...
        2*ones(1,size(datatrain2,2)),...
        3*ones(1,size(datatrain3,2))];
    
    datatest= [datatest1,datatest2,datatest3];
    dtest= [ones(1,size(datatest1,2)),...
        2*ones(1,size(datatest2,2)),...
        3*ones(1,size(datatest3,2))];
    % Normalization
    mu= mean(datatrain,2);
    sigma= std(datatrain')';
    datatrain= (datatrain- repmat(mu,1,size(datatrain,2)))./repmat(sigma,1,size(datatrain,2)) ;
    datatest= (datatest- repmat(mu,1,size(datatest,2)))./repmat(sigma,1,size(datatest,2)) ;
    %% step 2: train calssifier using train data & train label
    % voting : training
    mdl1= fitcknn(datatrain',dtrain,'NumNeighbors',3);
    mdl2= fitctree(datatrain',dtrain);
    mdl3= fitcdiscr(datatrain',dtrain);
    %% step 3:test trained classifier using test data
    P(1,:)= predict(mdl1,datatest');
    P(2,:)= predict(mdl2,datatest');
    P(3,:)= predict(mdl3,datatest');
    userlabel=unique(dtrain);
    output = myMejorityVoting(P,userlabel);
    %% step 4: validation
    C= confusionmat(dtest,output);
    Totalaccuracy(iter)=  sum(diag(C)) / sum(C(:)) *100;
    accuracy1(iter)= C(1,1) / sum(C(1,:)) *100;
    accuracy2(iter)= C(2,2) / sum(C(2,:)) *100;
    accuracy3(iter)= C(3,3) / sum(C(3,:)) *100;
    Ct= Ct+C;
end
Ct
disp(['total Accuracy: ',num2str(mean(Totalaccuracy)) ,'%'])
disp(['accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['accuracy3: ',num2str(mean(accuracy3)) ,'%'])
%% **************************Results******************* %%
% Ct =
% 
%     50     0     0
%      0    48     2
%      0     3    47
% 
% total Accuracy: 96.6667%
% accuracy1: 100%
% accuracy2: 96%
% accuracy3: 94%






