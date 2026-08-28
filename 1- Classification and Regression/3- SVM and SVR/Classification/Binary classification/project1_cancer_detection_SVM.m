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
load CancerDataset1
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
    
        %         kernel='rbf';
        %         kernelPar=[10];
        %         Constrain=100;
            
        %     kernel='polynomial';
        %     kernelPar=[3];
        %     Constrain=1000;
    
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
%% results
% total Accuracy: 96.381%
% sensitivity: 95.4965%
% specificity: 96.8658%


