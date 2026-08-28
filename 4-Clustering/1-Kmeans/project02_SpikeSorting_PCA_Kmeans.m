clc
clear 
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Neural Spike Sorting with Dimensionality Reduction
%  Algorithm       : PCA + K-Means Clustering – from scratch
%  
%  Description     : Spike sorting pipeline: First, Principal Component 
%                    Analysis (PCA) reduces spike waveforms from 60D to 2D, 
%                    then K-Means is applied for clustering in the 
%                    low-dimensional feature space.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/pattern-season08-clustering/
%  https://onlinebme.com/product/dimension-reduction-using-lda-pca/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background and video explanations, 
%  please visit OnlineBME.
%% ========================================================================
load Spikes
figure(1)
plot(Spikes,'b','linewidth',0.1)
hold on
grid on
grid minor
%% dimension reduction, feature extraction
m=2;
w=myPCA(Spikes,m);
features= w'*Spikes;
figure(2)
plot(features(1,:),features(2,:),'ob','linewidth',2)
grid on
grid minor
%%  clustering spikes, spike sorting 
k=3;
Nepoch=50;
% data= Spikes;
[labels,centers] = mykmeans(features,k,Nepoch);
%%
indx1= find(labels==1);
indx2= find(labels==2);
indx3= find(labels==3);

figure
plot(features(1,indx1),features(2,indx1),'ob','linewidth',2)
hold on
plot(features(1,indx2),features(2,indx2),'or','linewidth',2)
plot(features(1,indx3),features(2,indx3),'og','linewidth',2)
grid on
grid minor
plot(centers(1,:),centers(2,:),'xk','linewidth',2,'markersize',20)

figure
plot(Spikes(:,indx1),'b','linewidth',0.1)
hold on
plot(Spikes(:,indx2),'r','linewidth',0.1)
plot(Spikes(:,indx3),'g','linewidth',0.1)
grid on
grid minor




