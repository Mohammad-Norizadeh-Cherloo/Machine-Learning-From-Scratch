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
%  Algorithm       : G-Means Clustering – from scratch
%  
%  Description     : Application of G-Means for automatic determination 
%                    of the number of clusters in action potential 
%                    waveforms (spikes) in the original high-dimensional 
%                    space (length 60).
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
subplot(3,2,1:2)
plot(Spikes,'b','linewidth',0.1)
hold on
grid on
grid minor
%%  clustering spikes, spike sorting
% data= Spikes;
alpha=0.01;
[labels,centers] = myGmeans(Spikes,alpha);
%%
k=max(labels);
for i=1:k
    subplot(3,2,i+2)
    indxi= find(labels==i);
    plot(Spikes(:,indxi),'color',rand(1,3),'linewidth',0.1)
    hold on
    plot(centers(:,i),'-',LineWidth=3,Color=[0.5,0.5,0.5])
   
    grid on
    grid minor
end




