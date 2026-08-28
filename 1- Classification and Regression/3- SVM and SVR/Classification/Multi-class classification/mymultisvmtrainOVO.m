function [mdl] = mymultisvmtrainOVO(Xtrain,Ytrain,kernel,kernelPar,Constrain,flag)
%  Algorithm       : Multi-class SVM (One-vs-One) with Linear, RBF, and 
%                    Polynomial kernels – from scratch
%  
%  Description     : Manual implementation of Multi-class Support Vector 
%                    Machine using the One-vs-One (OvO) strategy.
%                    Includes Linear, RBF, and Polynomial kernels.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
userlabel= unique(Ytrain);
y1=userlabel(1);
y2=userlabel(2);
y3=userlabel(3);
%% 1 vs (2)
data1= Xtrain(:,Ytrain==userlabel(1));
data2= Xtrain(:,Ytrain==userlabel(2));
traindata=[data1,data2];
trainlabel= [y1*ones(1,size(data1,2)),y2*ones(1,size(data2,2))];

mdl.svm1 = mysvmtrain(traindata,trainlabel,kernel,kernelPar,Constrain,flag);
%% 1 vs (3)
data1= Xtrain(:,Ytrain==userlabel(1));
data2= Xtrain(:,Ytrain==userlabel(3));
traindata=[data1,data2];
trainlabel= [y1*ones(1,size(data1,2)),y3*ones(1,size(data2,2))];

mdl.svm2 = mysvmtrain(traindata,trainlabel,kernel,kernelPar,Constrain,flag);
%% 2 vs (3)
data1= Xtrain(:,Ytrain==userlabel(2));
data2= Xtrain(:,Ytrain==userlabel(3));
traindata=[data1,data2];
trainlabel= [y2*ones(1,size(data1,2)),y3*ones(1,size(data2,2))];

mdl.svm3 = mysvmtrain(traindata,trainlabel,kernel,kernelPar,Constrain,flag);
mdl.userlabel=userlabel;

end

