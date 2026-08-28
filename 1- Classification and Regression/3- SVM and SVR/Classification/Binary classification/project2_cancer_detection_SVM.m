clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Breast Cancer Classification (UCI Breast Cancer Dataset)
%  Algorithm       : Binary Support Vector Machine (SVM) - from scratch
%  
%  Description     : Manual implementation of Binary SVM (Hard-Margin and 
%                    Soft-Margin) with Linear, RBF, and 
%                    Polynomial kernels, including the dual formulation, quadratic 
%                    programming, and support vector identification.
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
load CancerDataset2
% Totaldata
% label
%% normalization
% normalize data to get mu=0 & sigma=1
mu= mean(Totaldata,2);
sigma= std(Totaldata')';
Totaldata= (Totaldata- repmat(mu,1,size(Totaldata,2)))./repmat(sigma,1,size(Totaldata,2)) ;
%% step 1: devide data into train(70%) and test(30%)
% using random sub-sampling validation method
div=0.7;
num= round(div* size(Totaldata,2));
for iter=1:100
    ind=randperm(size(Totaldata,2));
    Totaldata= Totaldata(:,ind);
    label= label(ind);
    
    datatrain= Totaldata(:,1:num);
    dtrain=  label(1:num);
    
    datatest= Totaldata(:,num+1:end);
    dtest= label(num+1:end);
    %% step 2: train calssifier using train data & train label
        kernel='linear';
        kernelPar=[];
        Constrain=100;
    %
    %     kernel='rbf';
    %     kernelPar=[8];
    %     Constrain=10;  
    %     kernel='polynomial';
    %     kernelPar=[3];
    %     Constrain=100;
    flag=0;
    [mdl] = mysvmtrain(datatrain,dtrain,kernel,kernelPar,Constrain,flag);
    %% step 3:test trained classifier using test data
    [output] = mysvmclassify(mdl,datatest);
    %% step 4: validation
    C= confusionmat(dtest,output);
    % % total accuracy
    accuracy(iter)= sum(diag(C)) / sum(C(:)) *100;
    % % accuracy 1
    sensitivity(iter)= C(1,1) / sum(C(1,:)) *100;
    % %  accuracy 2
    specificity(iter)= C(2,2) / sum(C(2,:)) *100;
    iter
end
disp(['total Accuracy: ',num2str(mean(accuracy)) ,'%'])
disp(['sensitivity: ',num2str(mean(sensitivity)) ,'%'])
disp(['specificity: ',num2str(mean(specificity)) ,'%'])
%%  linear results, C=100;
% total Accuracy: 95.3626%
% sensitivity: 94.0372%
% specificity: 96.1563%

%%  rbf results, C=100;sigma=10;
% total Accuracy: 96.9298%
% sensitivity: 94.5496%
% specificity: 98.4068%

%%  rbf results, C=10;sigma=8;
% total Accuracy: 97.5205%
% sensitivity: 94.8438%
% specificity: 99.1257%

%%  polynomial results, C=1000;d=3;
% total Accuracy: 94.8012%
% sensitivity: 92.7641%
% specificity: 95.9804%


