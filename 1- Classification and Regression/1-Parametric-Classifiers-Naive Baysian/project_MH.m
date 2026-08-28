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
%  Algorithm       : Naive Bayes Classifier (from scratch)
%  
%  Description     : Manual implementation of Mahalanobis Distance classifier 
%                    with full mathematical derivation and step-by-step coding.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + implementation):
%  https://onlinebme.com/product/pattern-parametric-classifiers/
%  https://onlinebme.com/product-category/machine-learning
%  
%  If you want the complete mathematical background, derivations, 
%  and video explanations of this algorithm, please visit OnlineBME.
%% ========================================================================,
load('fisheriris.mat')
%****** sestosa   [1-50]*******%
%****** vesicolor [51-100]*****%
%****** virginica [101-150]****%
for i=1:100
    %% step 1: devide data into train(70%) and test(30%)
    ind= randperm(size(iris,2));
    iris = iris(:,ind);
    label = label(ind);
    div= 0.7;
    num= round(div*size(iris,2));
    datatrain= iris(:,1:num);
    dtrain= label(1:num);
    %
    datatest= iris(:,num+1:end);
    dtest= label(num+1:end);
    %% step 2: train classifiers using train data & train label
    [MDL] = myMahalabobistrain(datatrain,dtrain);
    %% step 3: classify test data using trained classifier
    [predictedLables] =  myMahalabobisclassify(MDL,datatest);
    %% step 4: validation
    % Confusion matrix
    C= confusionmat(dtest,predictedLables);
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
%% **************Results********************%%
% total Accuracy: 97.2222%
% accuracy1: 100%
% accuracy2: 92.6928%
% accuracy3: 98.9262%


















