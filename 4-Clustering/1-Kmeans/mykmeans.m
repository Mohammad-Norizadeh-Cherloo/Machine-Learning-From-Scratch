function [labels,centers] = mykmeans(data,k,Nepoch)
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Function        : mykmeans
%  Algorithm       : K-Means Clustering – from scratch
%  
%  Description     : Manual implementation of the K-Means algorithm 
%                    with intelligent center initialization (farthest-point 
%                    based) and iterative assignment + update steps.
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

%% step 1: determine number of clusters & initialize centers
%% determine first center
ind= randperm(size(data,2));
centers(:,1)= data(:,ind(1:1));
if size(data,1)>1
    %% determine other centers
    N= size(data,2); % Number of samples
    for i= 2:k
        for j=1:size(centers,2)
            cj= centers(:,j);
            reptCj= repmat(cj,1,N);
            D(j,:)=  sqrt(sum((reptCj-data).^2));
        end
        D= prod(D);
        [mx,ind]= max(D);
        centers(:,i)= data(:,ind);
        D=[];
    end
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
    
elseif size(data,1)==1
    %% determine other centers
    N= size(data,2); % Number of samples
    for i= 2:k
        for j=1:size(centers,2)
            cj= centers(:,j);
            reptCj= repmat(cj,1,N);
            D(j,:)=  sqrt(((reptCj-data).^2));
        end
        D= prod(D);
        [mx,ind]= max(D);
        centers(:,i)= data(:,ind);
        D=[];
    end
    %
    N= size(data,2); % Number of samples
    for iter=1:Nepoch
        %% step 2:calculate distance between all samples and all centers
        for j=1:k
            cj= centers(:,j);
            reptCj= repmat(cj,1,N);
            dis(j,:)=  sqrt(((reptCj-data).^2));
        end
        %% step 3: update centers
        [mn,indx]= min(dis);
        for j=1:k
            xj= data(:,indx==j);
            centers(:,j)= mean(xj,2);
        end
    end    
end

%% step 4: clustering
labels= indx;


end

