clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Visualization of Fisher LDA Projection
%  Algorithm       : Fisher Linear Discriminant Analysis (FDA / LDA) – from scratch
%  
%  Description     : Demonstration of Fisher LDA projection direction on 
%                    synthetic 2D three-class data. Shows the optimal 
%                    projection line and the resulting 1D histograms of 
%                    the projected classes.
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
mu1=[-1 4];
mu2=[-7 -5];
mu3=[8 1];
sigma1=[2 0;0 2];
sigma2=[2 0;0 2];
sigma3=[3 0;0 3];
data1=mvnrnd(mu1,sigma1,100)';
data2=mvnrnd(mu2,sigma2,100)';
data3=mvnrnd(mu3,sigma3,100)';
% display dataset
plot(data1(1,:),data1(2,:),'bs','linewidth',2,...
    'markersize',15,'markerfacecolor',rand(1,3))
hold on
plot(data2(1,:),data2(2,:),'mo','linewidth',2,...
    'markersize',15,'markerfacecolor',rand(1,3))
plot(data3(1,:),data3(2,:),'y^','linewidth',1,...
    'markersize',15,'markerfacecolor',rand(1,3))
grid on
grid minor
axis([-15 15 -15 15])
%%
Totaldata=[data1,data2,data3];
group= [ones(1,size(data1,2)),2* ones(1,size(data2,2)),3* ones(1,size(data3,2))];
%%
m=1;
W = myFDA(Totaldata,group,m);
Xnew= W'*Totaldata;
x= -10:10;
s= W(2,1)/W(1,1);
y= s*x;
plot(x,y,'r','linewidth',3)

figure
indx1= find(group==1);
histogram(Xnew(indx1),50)
hold on
indx2= find(group==2);
histogram(Xnew(indx2),50)
indx3= find(group==3);
histogram(Xnew(indx3),50)
grid on
grid minor
