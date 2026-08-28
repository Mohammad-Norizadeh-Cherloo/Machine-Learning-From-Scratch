clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Classification using K-Nearest Neighbors (KNN)
%  Algorithm       : K-Nearest Neighbors Classifier (from scratch)
%  
%  Description     : Manual implementation of the classic KNN classifier 
%                    with distance metrics, k-selection, and full evaluation.
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
    predictedLables = mywknnclassify2(MDL,datatest);
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

%% result 1NN, Euclidean
% total Accuracy: 95.3333%
% accuracy1: 100%
% accuracy2: 92.5%
% accuracy3: 91.8095%
%% result KNN: K=5, Euclidean
% total Accuracy: 97.3333%
% accuracy1: 100%
% accuracy2: 96.6434%
% accuracy3: 97.3333%
%% result KNN: K=15, Euclidean
% total Accuracy: 97.3333%
% accuracy1: 100%
% accuracy2: 93.7778%
% accuracy3: 98.3333%

%% result KNN: K=45, Euclidean
% total Accuracy: 92%
% accuracy1: 100%
% accuracy2: 92.1429%
% accuracy3: 85.9596%
%% result KNN: K=5, cityblock
% total Accuracy: 92.6667%
% accuracy1: 100%
% accuracy2: 90.4747%
% accuracy3: 89.3939%

%% result KNN: K=5, chebychev
% total Accuracy: 97.3333%
% accuracy1: 100%
% accuracy2: 97.0833%
% accuracy3: 96.3636%

%% result KNN: K=5, minkovsky
% total Accuracy: 97.3333%
% accuracy1: 100%
% accuracy2: 92.4206%
% accuracy3: 98.4615%

%% result KNN: K=5, cosine
% total Accuracy: 96.6667%
% accuracy1: 100%
% accuracy2: 92.4786%
% accuracy3: 98.5714%

%% result KNN: K=5, correlation
% total Accuracy: 97.3333%
% accuracy1: 100%
% accuracy2: 92.6282%
% accuracy3: 98%
