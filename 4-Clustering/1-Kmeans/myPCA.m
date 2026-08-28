function W=myPCA(Xtrain,m)
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Function        : myPCA
%  Algorithm       : Principal Component Analysis (PCA) – from scratch
%  
%  Description     : Manual implementation of Principal Component Analysis 
%                    for unsupervised dimensionality reduction.
%                    Steps: mean centering, covariance matrix computation, 
%                    eigenvalue decomposition, and selection of the top-m 
%                    eigenvectors.
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

% pca function
% Xtrain is a d*N matrix
% m is the number of igenvectors
% step 1: means  zeros
meanX=mean(Xtrain');
Xtrain=Xtrain-repmat(meanX',1,size(Xtrain,2));
% plot(Xtrain(1,:),Xtrain(2,:),'r.','DisplayName','Xtrain')
% step 2: calculatin covariance matrix
C=cov(Xtrain');
% step 3: diagnalization
[U,V]=eig(C);
V=diag(V);
% step 4: sorting data
[V,ind]=sort(V,'descend');
U=U(:,ind);
W=U(:,1:m);
end

