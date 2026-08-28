function [group] = myMahalabobisclassify(MDL,datatest)
%Manual implementation of Mahalabobis classifier 
muP=MDL.mu;
labels=MDL.Y;
sigmaP=MDL.sigma;
for i=1:size(datatest,2)
    xi= datatest(:,i);
    for j= 1:size(muP,2)
        dis(j)= sqrt( (xi-muP(:,j))'* inv(sigmaP(:,:,j)) * (xi-muP(:,j)));
    end
    [m,ind]= min(dis);
    group(i)= labels(ind);
end
end