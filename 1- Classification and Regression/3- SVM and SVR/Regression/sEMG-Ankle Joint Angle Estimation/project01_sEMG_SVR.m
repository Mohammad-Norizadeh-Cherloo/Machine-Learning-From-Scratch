clc;
clear;
close all;
%% ========================================================================
%  Author          : Mohammad Norizadeh Cherloo
%  Website         : https://onlinebme.com/
%  GitHub          : https://github.com/Mohammad-Norizadeh-Cherloo
%  Google Scholar  : https://scholar.google.com/citations?user=fIKpYm8AAAAJ
%  
%  Project         : Ankle Joint Angle Estimation from 3-channel sEMG
%  Algorithm       : Support Vector Regression (SVR) with Linear, RBF, 
%                    and Polynomial kernels – from scratch
%  
%  Description     : Manual implementation of Support Vector Regression 
%                    (SVR) using Linear, RBF, and Polynomial kernels.
%                    Applied to 3-electrode sEMG signals for continuous 
%                    estimation of ankle joint angle.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
%  
%  Full video course (theory + equations + step-by-step implementation):
%  https://onlinebme.com/product/svm/
%  https://onlinebme.com/product-category/machine-learning
%  
%  For the complete mathematical background, derivations, and video 
%  explanations of this algorithm, please visit OnlineBME.
%% ========================================================================
load('EMG_Data_Train.mat')
load('Label_Train_1Deg.mat')
%% feature extraction
c=0;
for i=500:200:size(EMG_Data_Train,1)
    c=c+1;
    ind= i-499:i;
    trial= EMG_Data_Train(ind,:);
    data(:,c)= [mean(trial),var(trial),rms(trial)]';
    output(c)= Label_Train_77_Class_1Deg(i);
end
%% step 1: devide data into train and test
div=0.7;
num=round(div*size(data,2));
datatrain=data(:,1:num);
dtrain=output(1:num);

datatest=data(:,num+1:end);
dtest=output(num+1:end);

%% normalization
mu=mean(datatrain,2);
sigma=std(datatrain')';
datatrain=(datatrain-repmat(mu,1,size(datatrain,2)))...
    ./repmat(sigma,1,size(datatrain,2));
datatest=(datatest-repmat(mu,1,size(datatest,2)))...
    ./repmat(sigma,1,size(datatest,2));
%% step 2: train model using traindata & train output
kernel='linear';
kernelPar=[];
C=100;
epsilon=0.1;
% kernel='rbf';
% kernelPar=10;
% C=1000;
% epsilon=0.1;
% kernel='polynomial';
% kernelPar=2;
% C=1000;
% epsilon=0.8;
mdl = mySVRtrain(datatrain,dtrain,kernel,kernelPar,C,epsilon);
%% step 3: predict test outputs using trained model
output = mySVRpredict(mdl,datatest);
%% step 4: validation
figure
plot(dtest,'b','linewidth',1);
hold on
plot(output,'r','linewidth',1);
legend('desired output','SVR predicted output')
cr= corr(dtest',output');
title (['correlation: ',num2str(cr)])
