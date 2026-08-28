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
%  Algorithm       : Stacking Ensemble – from scratch
%  
%  Description     : Manual implementation of Stacking Ensemble 
%                    (base learners + meta-learner) on the UCI Breast Cancer dataset.
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
load CancerDataset2
% Totaldata
% label
%% step 1: devide data into train(70%) and test(30%)
div=0.7;
num= round(div* size(Totaldata,2));
datatrain= Totaldata(:,1:num);
dtrain=  label(1:num);

datatest= Totaldata(:,num+1:end);
dtest= label(num+1:end);
%% normalization
% normalize data to get mu=0 & sigma=1
mu=mean(datatrain,2);
sigma=std(datatrain')';
datatrain=(datatrain-repmat(mu,1,size(datatrain,2)))...
    ./repmat(sigma,1,size(datatrain,2));
datatest=(datatest-repmat(mu,1,size(datatest,2)))...
    ./repmat(sigma,1,size(datatest,2));

%% 
div=0.7;
num= round(div* size(datatrain,2));
datavalid= datatrain(:,num+1:end);
dvalid= dtrain(num+1:end);

datatrain= datatrain(:,1:num);
dtrain=  dtrain(1:num);
%% step 2: train calssifier using train data & train label
% stacking : first level training
mdl1= fitcknn(datatrain',dtrain,'NumNeighbors',3);
mdl2= fitctree(datatrain',dtrain);
mdl3= fitcdiscr(datatrain',dtrain);
mdl4= fitcsvm(datatrain',dtrain,'Standardize',1);
mdl5= fitcsvm(datatrain',dtrain,'Standardize',1,'KernelFunction','polynomial');
mdl6= fitcnb(datatrain',dtrain);

%% second level training 
Xtrain(1,:)= predict(mdl1,datavalid');
Xtrain(2,:)= predict(mdl2,datavalid');
Xtrain(3,:)= predict(mdl3,datavalid');
Xtrain(4,:)= predict(mdl4,datavalid');
Xtrain(5,:)= predict(mdl5,datavalid');
Xtrain(6,:)= predict(mdl6,datavalid');

mdl = mySLPtrainR(Xtrain,dvalid);

%% step 3:test trained classifier using test data
P(1,:)= predict(mdl1,datatest');
P(2,:)= predict(mdl2,datatest');
P(3,:)= predict(mdl3,datatest');
P(4,:)= predict(mdl4,datatest');
P(5,:)= predict(mdl5,datatest');
P(6,:)= predict(mdl6,datatest');
%% weighted voting : combine votes
output = mySLPtestR(mdl,P);
output = round(output); 
%% step 4: validation
C= confusionmat(dtest,output)
% % total accuracy
accuracy= sum(diag(C)) / sum(C(:)) *100;
sensitivity= C(1,1) / sum(C(1,:)) *100;
specificity= C(2,2) / sum(C(2,:)) *100;

disp(['total Accuracy: ',num2str(accuracy) ,'%'])
disp(['sensitivity: ',num2str(sensitivity) ,'%'])
disp(['specificity: ',num2str(specificity) ,'%'])
%% **************************Results******************* %%
% C =
% 
%     39     0
%      2   130
% 
% total Accuracy: 98.8304%
% sensitivity: 100%
% specificity: 98.4848%

