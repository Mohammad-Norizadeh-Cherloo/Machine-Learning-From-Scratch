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
%  Algorithm       : PCA + G-Means Clustering – from scratch
%  
%  Description     : Spike sorting pipeline: First, Principal Component 
%                    Analysis (PCA) reduces spike waveforms from 60D to 
%                    10D, then G-Means is applied to automatically 
%                    discover the number of clusters in the reduced 
%                    feature space.
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
m=10;
w=myPCA(Spikes,m);
features= w'*Spikes;
% figure(2)
% plot(features(1,:),features(2,:),'ob','linewidth',2)
% grid on
% grid minor
%%  clustering spikes, spike sorting 
alpha=0.01;
[labels,centers] = myGmeans(features,alpha);
%%
k=max(labels);
for i=1:k
    subplot(2,2,i)
    indxi= find(labels==i);
    plot(Spikes(:,indxi),'color',rand(1,3),'linewidth',0.1)
    grid on
    grid minor
end


