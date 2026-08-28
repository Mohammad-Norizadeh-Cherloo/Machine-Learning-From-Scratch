clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Classification using Weighted K-Nearest Neighbors (WKNN)
%  Algorithm       : Weighted KNN Classifier (from scratch)
%  
%  Description     : Manual implementation of Weighted KNN (including 
%                    distance-based weighting schemes from research papers).
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/k-nearest-neighbors/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background and video explanations, 
%  please visit OnlineBME.
%% ========================================================================
load('fisheriris.mat')
%****** sestosa   [1-50]*******%
%****** vesicolor [51-100]*****%
%****** virginica [101-150]****%
%% step 1: devide data into train and test
% using k-fold cross validation method
ind= randperm(size(iris,2));
iris = iris(:,ind);
label = label(ind);
K=5;
fold= floor(size(iris,2) / K);
for i=1:K
    indtest= (i-1)*fold+1 : i*fold;
    indtrain= 1:size(iris,2);
    indtrain(indtest)=[];
    
    datatrain= iris(:,indtrain);
    dtrain= label(indtrain);
    
    datatest= iris(:,indtest);
    dtest= label(indtest);
    %% step 2: train calssifier using train data & train label
    % Distance: Euclidean ,Cityblock,Chebychev,Minkowski,Cosine,Correlation
    kn=5;
    Distance='Euclidean';
    MDL = myknntrain(datatrain,dtrain,kn,Distance);
    %% step 3: classify test data using trained classifier
    predictedLables = mywknnclassify5(MDL,datatest);
    %% step 4: validation
    % Confusion matrix
    C= confusionmat(dtest,predictedLables');
    % total accuracy
    accuracy(i)=  sum(diag(C)) / sum(C(:)) *100;
    % % accuracy 1
    accuracy1(i)= C(1,1) / sum(C(1,:)) *100;
    %  accuracy 2
    accuracy2(i)= C(2,2) / sum(C(2,:)) *100;
    %  accuracy 3
    accuracy3(i)= C(3,3) / sum(C(3,:)) *100;
end
disp(['total Accuracy: ',num2str(mean(accuracy)) ,'%'])
disp(['accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['accuracy3: ',num2str(mean(accuracy3)) ,'%'])

%% result wknn1 w= 1/D
% total Accuracy: 97.3333%
% accuracy1: 100%
% accuracy2: 96.1818%
% accuracy3: 96.3333%

%% result wknn1 w= 1/D.^2
% total Accuracy: 96.6667%
% accuracy1: 100%
% accuracy2: 93%
% accuracy3: 95.7778%

%% result wknn1 w= 1/ (c+D.^2)
% total Accuracy: 96%
% accuracy1: 100%
% accuracy2: 91.9192%
% accuracy3: 96.4615%

%% result wknn1 w= exp (- D.^2 / simga^2)
% total Accuracy: 97.3333%
% accuracy1: 100%
% accuracy2: 96.3636%
% accuracy3: 96.3492%

%% result wknn1 w= 1/D ,D= D / D(k+1)
% total Accuracy: 96%
% accuracy1: 100%
% accuracy2: 91.7143%
% accuracy3: 98.1818%



