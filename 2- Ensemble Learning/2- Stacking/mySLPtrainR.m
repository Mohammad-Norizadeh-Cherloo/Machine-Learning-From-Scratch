function [mdl] = mySLPtrainR(Xtrain,Ytrain)
%  Description     : Manual implementation of Stacking Ensemble 
%                    (meta-learner) on the UCI Breast Cancer dataset.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.

%% train
Xtrain= [-ones(1,size(Xtrain,2));Xtrain];
X= Xtrain;
d= Ytrain;
Rx= X*X'; % autocorrelation
Rdx= (d)*X'; % cross corrlation

w=  Rdx* inv(Rx+ (eye(size(Rx))+eps));
mdl.w=w;
end