function [mdl] = mymultisvmtrainOVR(Xtrain,Ytrain,kernel,kernelPar,Constrain,flag)
%  Algorithm       : Multi-class SVM (One-vs-Rest) with Linear, RBF, and 
%                    Polynomial kernels – from scratch
%  
%  Description     : Manual implementation of Multi-class Support Vector 
%                    Machine using the One-vs-Rest (OvR) strategy.
%                    Includes Linear, RBF, and Polynomial kernels.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
userlabel= unique(Ytrain);
%% 1 vs (2,3)
data1= Xtrain(:,Ytrain==userlabel(1));
data2= Xtrain(:,Ytrain~=userlabel(1));
traindata=[data1,data2];
trainlabel= [ones(1,size(data1,2)),2*ones(1,size(data2,2))];

mdl.svm1 = mysvmtrain(traindata,trainlabel,kernel,kernelPar,Constrain,flag);
%% 2 vs (1,3)
data1= Xtrain(:,Ytrain==userlabel(2));
data2= Xtrain(:,Ytrain~=userlabel(2));
traindata=[data1,data2];
trainlabel= [ones(1,size(data1,2)),2*ones(1,size(data2,2))];

mdl.svm2 = mysvmtrain(traindata,trainlabel,kernel,kernelPar,Constrain,flag);
%% 3 vs (1,2)
data1= Xtrain(:,Ytrain==userlabel(3));
data2= Xtrain(:,Ytrain~=userlabel(3));
traindata=[data1,data2];
trainlabel= [ones(1,size(data1,2)),2*ones(1,size(data2,2))];

mdl.svm3 = mysvmtrain(traindata,trainlabel,kernel,kernelPar,Constrain,flag);
mdl.userlabel=userlabel;

end

