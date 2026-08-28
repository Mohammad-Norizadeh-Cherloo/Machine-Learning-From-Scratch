function [MDL] = myknntrain(datatrain,dtrain,k,Distance)
% KNN Classifier (from scratch)
MDL.datatrain=datatrain;
MDL.dtrain=dtrain;
MDL.NumNeighbors=k;
MDL.Distance=Distance;
Y= unique(dtrain);
MDL.Y= Y;
end

