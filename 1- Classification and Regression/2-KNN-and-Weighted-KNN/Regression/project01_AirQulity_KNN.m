clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Air Quality Prediction (UCI Air Quality Dataset)
%  Algorithm       : K-Nearest Neighbors Regressor (from scratch)
%  
%  Description     : Manual implementation of KNN for regression on the 
%                    UCI Air Quality dataset. Includes different distance 
%                    metrics and k-selection analysis.
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
load('AirQualityUCI.mat')
output=AirQualityUCI(:,5)';
ind=1:14;
ind(5)=[];
data=AirQualityUCI(:,ind)';
%% deviding data into training data and test data
ind= randperm(size(data,2));
data = data(:,ind);
output = output(ind);

div=0.9;
num=round(div*size(data,2));
datatrain=data(:,1:num);
dtrain=output(1:num);

datatest=data(:,num+1:end);
dtest=output(num+1:end);

%% normalization
mu=mean(datatrain,2);
sigma=std(datatrain')';
datatrain=(datatrain-repmat(mu,1,size(datatrain,2)))...
    ./repmat(sigma,1,size(datatrain,2));
datatest=(datatest-repmat(mu,1,size(datatest,2)))...
    ./repmat(sigma,1,size(datatest,2));
%% step 2: train knn using traindata & trainlabel
% Distance: Euclidean ,Cityblock,Chebychev,Minkowski,Cosine,Correlation
k=5;
Distance='Euclidean';
MDL = myknntrainR(datatrain,dtrain,k,Distance);
%% step 3: predict test outputs using trained model
predicted_output= myknnregression(MDL,datatest);
%% validation
figure
plot(dtest,'b','linewidth',1);
hold on
plot(predicted_output,'r','linewidth',1);
legend('desired output','knn predicted output')
cr= corr(dtest',predicted_output');
title (['correlation: ',num2str(cr)])







