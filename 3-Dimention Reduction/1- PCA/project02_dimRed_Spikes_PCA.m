clc
clear
close all
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Dimensionality Reduction of Neural Action Potentials (Spikes)
%  Algorithm       : Principal Component Analysis (PCA) – from scratch
%  
%  Description     : Manual implementation of PCA for dimensionality reduction 
%                    of action potential waveforms. Spikes of length 60 
%                    (60-dimensional time-domain vectors) are projected 
%                    into 2D space for visualization and analysis.
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
load Spikes
plot(Spikes,'b','linewidth',1)
grid on
grid minor
%% dimension reduction
% PCA
m=2;
[W] = mypca(Spikes,m);
% step 6: map data using m eigen vectors
Features= W'*Spikes;
figure
plot(Features(1,:),Features(2,:),'ko','linewidth',1.5,...
    'markersize',10,'markerfacecolor',rand(1,3))
grid on
grid minor

