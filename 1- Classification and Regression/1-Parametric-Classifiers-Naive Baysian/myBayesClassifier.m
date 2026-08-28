function [group] = myBayesClassifier(data,muP,sigmaP,pw)
% Manual implementation of Naive Bayes classifier 
for i=1:size(data,2)
    xi= data(:,i);
    for j= 1:size(muP,2)
        L(j) =  mypdf(xi,muP(:,j),sigmaP(:,:,j));
        ps(j) = L(j) * pw(j);
    end
    [~,group(i)] = max(ps);
end

end

