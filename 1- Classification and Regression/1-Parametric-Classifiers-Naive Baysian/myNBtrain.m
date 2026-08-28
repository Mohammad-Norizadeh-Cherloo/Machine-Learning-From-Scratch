function [MDL] = myNBtrain(datatrain,dtrain,pw)
% Manual implementation of Naive Bayes classifier 
labels= unique(dtrain);
numClass= numel(labels);
for i=1:numClass
    ind= find(dtrain==labels(i));
    mu(:,i) = mean(datatrain(:,ind),2);
    sigma(:,:,i) = cov(datatrain(:,ind)');
end
MDL.mu= mu;
MDL.sigma= sigma;
MDL.pw= pw;
MDL.Y=labels;
end

