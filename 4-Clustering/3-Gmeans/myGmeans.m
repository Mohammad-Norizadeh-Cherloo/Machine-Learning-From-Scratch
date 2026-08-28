function [labels,centers] = myGmeans(data,alpha)
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Function        : myGmeans
%  Algorithm       : G-Means Clustering – from scratch
%  
%  Description     : Manual implementation of the G-Means algorithm 
%                    (Hamerly & Elkan). Automatically determines the 
%                    number of clusters by repeatedly testing whether 
%                    data in each cluster follows a Gaussian distribution 
%                    (Anderson-Darling test) and splitting non-Gaussian 
%                    clusters.
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

centers= mean(data,2);
N= size(data,2);
labels= ones(1,N);
%%
Nepoch=10;
thr=2;
while thr>0
    C_kids=[];
    C_parents=[];
    for i=1:size(centers,2)
        ci= centers(:,i);
        %% step 1: determine significance level,Alpha
        Xi= data(:,labels==i);
        Cv= cov(Xi');
        [U,D]= eig(Cv);
        [D,indx]= sort(diag(D),'descend');
        landa= D(1);
        s= U(:,indx(1));
        m= s* sqrt(2*landa / pi);
        c1= ci+m;
        c2= ci-m;
        children = [c1,c2];
        [label,children] = mykmeans(Xi,children,Nepoch);
        c1= children(:,1);
        c2= children(:,2);
        %% calculate V
        v= c1-c2;
        %% project data on v
        Xprime= (v'*Xi) / (norm(v));
        %% normalize data
        mu= mean(Xprime);
        sd= std(Xprime);
        Xprime= (Xprime- mu) / sd;
        h(i)= adtest(Xprime,'Alpha',alpha);
        if h(i)==1
            C_kids=[C_kids,children];
            
        elseif h(i)==0
            C_parents=[C_parents,ci];
        end
    end
    thr= sum(h);
    if thr>0
        num= size(C_kids,2)/2+1;
        indx= randperm(size(C_kids,2));
        
        C=[C_parents,C_kids(:,indx(1:num))];
        [labels,centers] = mykmeans(data,C,Nepoch);
    end
end

end
function [labels,centers] = mykmeans(data,centers,Nepoch)
%% step 1: determine number of clusters & initialize centers
k= size(centers,2);
%
N= size(data,2); % Number of samples
for iter=1:Nepoch
    %% step 2:calculate distance between all samples and all centers
    for j=1:k
        cj= centers(:,j);
        reptCj= repmat(cj,1,N);
        dis(j,:)=  sqrt(sum((reptCj-data).^2));    
    end
    %% step 3: update centers
    [mn,indx]= min(dis);
    for j=1:k
        xj= data(:,indx==j);
        centers(:,j)= mean(xj,2);
    end
end
%% step 4: clustering 
labels= indx;
end


