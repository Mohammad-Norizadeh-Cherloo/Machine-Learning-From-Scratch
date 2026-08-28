
clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Dimensionality Reduction on IRIS Dataset
%  Algorithm       : Fisher Linear Discriminant Analysis (FDA / LDA) – from scratch
%  
%  Description     : Application of Fisher LDA to reduce the IRIS dataset 
%                    from 4D to 2D and visualize the projected classes.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/dimension-reduction-using-lda-pca/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background, derivations, and video 
%  explanations of this algorithm, please visit OnlineBME.
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
% display dataset
% axis([0 10 -5 5])
%%
m=2;
W = myFDA(iris,label,m);
iris= W'*iris;

plot(iris(1,label==1),iris(2,label==1),'bs','linewidth',2,...
    'markersize',15,'markerfacecolor',rand(1,3))
hold on
plot(iris(1,label==2),iris(2,label==2),'mo','linewidth',2,...
    'markersize',15,'markerfacecolor',rand(1,3))
plot(iris(1,label==3),iris(2,label==3),'y^','linewidth',1,...
    'markersize',15,'markerfacecolor',rand(1,3))
grid on
grid minor












