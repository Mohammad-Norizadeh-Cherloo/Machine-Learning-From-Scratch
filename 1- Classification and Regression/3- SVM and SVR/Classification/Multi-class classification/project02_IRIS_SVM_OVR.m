clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Multi-class Classification on IRIS Dataset
%  Algorithm       : Multi-class SVM (One-vs-Rest) with Linear, RBF, and 
%                    Polynomial kernels – from scratch
%  
%  Description     : Manual implementation of Multi-class Support Vector 
%                    Machine using the One-vs-Rest (OvR) strategy.
%                    Includes Linear, RBF, and Polynomial kernels.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/svm/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background, derivations, and video 
%  explanations of this algorithm, please visit OnlineBME.
%% ========================================================================
load('fisheriris.mat')
%****** sestosa   [1-50]*******%
%****** vesicolor [51-100]*****%
%****** virginica [101-150]****%
%% normalization
% normalize data to get mu=0 & sigma=1
mu= mean(iris,2);
sigma= std(iris')';
iris= (iris- repmat(mu,1,size(iris,2)))./repmat(sigma,1,size(iris,2)) ;
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
    
    traindata1= data1(:,indtrain);
    testdata1= data1(:,indtest);
    
    traindata2= data2(:,indtrain);
    testdata2= data2(:,indtest);
    
    traindata3= data3(:,indtrain);
    testdata3= data3(:,indtest);
    
    traindata= [traindata1,traindata2,traindata3];
    trainlabel= [ones(1,size(traindata1,2)),...
        2*ones(1,size(traindata2,2)),...
        3*ones(1,size(traindata3,2))];
    
    testdata= [testdata1,testdata2,testdata3];
    testlabel= [ones(1,size(testdata1,2)),...
        2*ones(1,size(testdata2,2)),...
        3*ones(1,size(testdata3,2))];
    %% step 2: train calssifier using train data & train label
    %     kernel='linear';
    %     kernelPar=[];
    %     Constrain=100;
    kernel='rbf';
    kernelPar=[8];
    Constrain=10;
    flag=0;
    [mdl] = mymultisvmtrainOVR(traindata,trainlabel,kernel,kernelPar,Constrain,flag);
    
    %% step 3:test trained classifier using test data
    [output] = mymultisvmclassifyOVR(mdl,testdata);
    %% step 4: validation
    C= confusionmat(testlabel,output);
    % total accuracy
    accuracy(iter)=  sum(diag(C)) / sum(C(:)) *100;
    % % accuracy 1
    accuracy1(iter)= C(1,1) / sum(C(1,:)) *100;
    %  accuracy 2
    accuracy2(iter)= C(2,2) / sum(C(2,:)) *100;
    %  accuracy 3
    accuracy3(iter)= C(3,3) / sum(C(3,:)) *100;
    Ct= Ct+C;
end
Ct
disp(['total Accuracy: ',num2str(mean(accuracy)) ,'%'])
disp(['accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['accuracy3: ',num2str(mean(accuracy3)) ,'%'])




