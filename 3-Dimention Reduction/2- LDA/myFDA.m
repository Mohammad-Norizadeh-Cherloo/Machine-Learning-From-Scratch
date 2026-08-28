function [W] = myFDA(datatrain,dtrain,m)
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Function        : myFDA
%  Algorithm       : Fisher Linear Discriminant Analysis (FDA / LDA) – from scratch
%  
%  Description     : Manual implementation of Fisher Discriminant Analysis 
%                    for supervised dimensionality reduction.
%                    Computes within-class (SW) and between-class (SB) 
%                    scatter matrices and solves the generalized eigenvalue problem.
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
% m<= C-1, C= number of class
N= size(datatrain,2);
d= size(datatrain,1);
mt= mean(datatrain,2);

SW= zeros(d,d);
SB= zeros(d,d);
% 
userlabel= unique(dtrain);
C= numel(userlabel);%number of class
for i=1:C
    ind= find(dtrain==userlabel(i));
    Xc= datatrain(:,ind);
    si= cov(Xc');
    mi= mean(Xc,2);
    ni= numel(ind)/ N;
    
    SW= SW+si;
    SB= SB+ ( ni * (mi-mt)*(mi-mt)'); 
end
%% step 4: eigen value decomposition
[U,D]= eig(SB,SW);
% [U,D]= eig(inv(SW)*SB);
%% step 5: sort eignen vectors according to eigen valuse
D= diag(D);
[D,ind]= sort(D,'descend');
U= U(:,ind);
%% step 6: select best eigen vectors
W= U(:,1:m);
end

