function [MDL] = myknntrainR(datatrain,dtrain,k,Distance)
% K-Nearest Neighbors Regressor (from scratch)
%Distance= Euclidean ,Cityblock,Chebychev,Minkowski,Cosine,Correlation
MDL.datatrain=datatrain;
MDL.dtrain=dtrain;
MDL.NumNeighbors=k;
MDL.Distance=Distance;
end

