function [MDL] = myEuqlideantrain(datatrain,dtrain)
%Manual implementation of Euqlidean classifier 
labels= unique(dtrain);
numClass= numel(labels);
for i=1:numClass
    ind= find(dtrain==labels(i));
    mu(:,i) = mean(datatrain(:,ind),2);
end
MDL.mu= mu;
MDL.Y=labels;
end
