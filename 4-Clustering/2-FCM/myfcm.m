function [C,U] = myfcm(data,k,Nepoch)
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Function        : myfcm
%  Algorithm       : Fuzzy C-Means (FCM) Clustering – from scratch
%  
%  Description     : Manual implementation of Fuzzy C-Means algorithm 
%                    with fuzzifier m=2. Updates cluster centers and 
%                    membership matrix iteratively.
%                    Supports both multi-dimensional and 1D data.
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

%% step 1: initialize membership functions, determine number of culsters
N= size(data,2);
d= size(data,1);
Mu= rand(k,N);
centers= zeros(d,k);
m= 2; % fuzzier
if d>1
    for iter= 1:Nepoch
        %    c
        %    u
        %% step 2: update centers
        num=((Mu.^m) * data')';
        denum= sum(Mu.^m,2)';
        D= repmat(denum,d,1);
        centers= num./(D+eps);
        %% step 3: update membership function
        for j=1:k
            cj= centers(:,j);
            reptCj= repmat(cj,1,N);
            dis(j,:)=  sqrt(sum((reptCj-data).^2));
        end
        for j=1:k
            disj= dis(j,:);
            Dj= repmat(disj,k,1);
            denumM=(Dj./dis).^(2/(m-1));
            Dnum=sum(denumM);
            Mu(j,:)= 1./(Dnum+eps);
        end
    end
elseif d==1
    for iter= 1:Nepoch
        %    c
        %    u
        %% step 2: update centers
        num=((Mu.^m) * data')';
        denum= sum(Mu.^m,2)';
        D= repmat(denum,d,1);
        centers= num./(D+eps);
        %% step 3: update membership function
        for j=1:k
            cj= centers(:,j);
            reptCj= repmat(cj,1,N);
            dis(j,:)=  sqrt(((reptCj-data).^2));
        end
        for j=1:k
            disj= dis(j,:);
            Dj= repmat(disj,k,1);
            denumM=(Dj./dis).^(2/(m-1));
            Dnum=sum(denumM);
            Mu(j,:)= 1./(Dnum+eps);
        end
    end
end
C= centers;
U= Mu;
end

