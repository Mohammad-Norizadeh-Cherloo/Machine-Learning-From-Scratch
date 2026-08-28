function [output] = mySLPtestR(mdl,datatest)
%  Description     : Manual implementation of Stacking Ensemble 
%                    (meta-learner) on the UCI Breast Cancer dataset.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.

w=mdl.w;
%% test
datatest= [-ones(1,size(datatest,2));datatest];
%
output= w*datatest;

end

