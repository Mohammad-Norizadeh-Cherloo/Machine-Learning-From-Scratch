clc
clear 
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Neural Spike Sorting
%  Algorithm       : K-Means Clustering – from scratch
%  
%  Description     : Application of K-Means for clustering action potential 
%                    waveforms (spikes) directly in the original 
%                    high-dimensional space (length 60).
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/pattern-season08-clustering/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background and video explanations, 
%  please visit OnlineBME.
%% ========================================================================
load Spikes
figure
plot(Spikes,'b','linewidth',0.1)
hold on
grid on
grid minor
%%  clustering spikes, spike sorting 
k=3;
Nepoch=50;
% data= Spikes;
[labels,centers] = mykmeans(Spikes,k,Nepoch);
%%
indx1= find(labels==1);
indx2= find(labels==2);
indx3= find(labels==3);
figure
subplot(2,3,[1 3])
plot(Spikes,'k','linewidth',0.1)
grid on
grid minor

subplot(2,3,4)
plot(Spikes(:,indx1),'b','linewidth',0.1)
grid on
grid minor
hold on
plot(centers(:,1),'k','linewidth',3)


subplot(2,3,5)
plot(Spikes(:,indx2),'r','linewidth',0.1)
grid on
grid minor
hold on
plot(centers(:,2),'k','linewidth',3)

subplot(2,3,6)
plot(Spikes(:,indx3),'g','linewidth',0.1)
grid on
grid minor
hold on
plot(centers(:,3),'k','linewidth',3)




