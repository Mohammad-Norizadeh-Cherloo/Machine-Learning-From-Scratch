function [group] = myEuqlideanclassify(MDL,datatest)
%Manual implementation of Euqlidean classifier 
muP=MDL.mu;
labels=MDL.Y;
for i=1:size(datatest,2)
    xi= datatest(:,i);
    for j= 1:size(muP,2)
        dis(j) = sqrt((xi-muP(:,j))'*(xi-muP(:,j)));
    end
    [m,ind]= min(dis);
    group(i)= labels(ind);
end
end