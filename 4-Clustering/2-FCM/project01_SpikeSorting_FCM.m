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
%  Algorithm       : Fuzzy C-Means (FCM) Clustering – from scratch
%  
%  Description     : Application of Fuzzy C-Means for soft clustering of 
%                    action potential waveforms (spikes) in the original 
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
[C,U] = myfcm(Spikes,k,Nepoch);
%%
[mx,labels]= max(U);
%%
indx1= find(labels==1);
indx2= find(labels==2);
indx3= find(labels==3);
figure
plot(Spikes(:,indx1),'b','linewidth',0.1)
hold on
plot(Spikes(:,indx2),'r','linewidth',0.1)
plot(Spikes(:,indx3),'g','linewidth',0.1)
grid on
grid minor




