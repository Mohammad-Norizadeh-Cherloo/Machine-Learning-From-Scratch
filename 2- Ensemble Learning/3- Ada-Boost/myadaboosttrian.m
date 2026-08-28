function [mdl] = myadaboosttrian(datatrain,dtrain,M,WL,learnerPar)
%  Algorithm       : AdaBoost (Adaptive Boosting) – from scratch
%  
%  Description     : Manual implementation of AdaBoost algorithm 
%                    (weight updating, weak learners combination) on the Iris dataset.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.

% WL weak learner: 'slp','svm','lda','knn','tree'
userlabel= unique(dtrain);
%% trianing
% train weak learners using traind data & labels
numModels=M;
dataset=[datatrain;dtrain];
% step 1: initialize training data weights
N= size(datatrain,2);
w=ones(1,N)/N;
D=w;

for m= 1:numModels
    % generate dataset using data distribution
    p_min=min(D);
    p_max=max(D);
    Dt=[];
    p = (p_max-p_min)*rand(1) + p_min;
    for i=1:length(D)
        if D(i)>=p
            d(:,i)=dataset(:,i);
        end
        t=randi(size(d,2));
        Dt=[Dt ,d(:,t)];
    end
    datatrain_bs= Dt(1:end-1,:);
    dtrainbs= Dt(end,:);
    % step 2-a:  fit model on training data using wieghts
    switch WL
        case 'slp'
            % step 2-a:  fit model on training data using wieghts
            models{m} = mySLPtrain(datatrain_bs,dtrainbs);
            % test traind madel(ith model) using all train data
            output = mySLPtest(models{m},datatrain);
        case 'svm'
            % step 2-a:  fit model on training data using wieghts
            models{m} = fitcsvm(datatrain_bs',dtrainbs,'Standardize',1);
            % test traind madel(ith model) using all train data
            output = predict(models{m},datatrain')';
        case 'lda'
            % step 2-a:  fit model on training data using wieghts
            models{m} = fitcdiscr(datatrain_bs',dtrainbs);
            % test traind madel(ith model) using all train data
            output = predict(models{m},datatrain')';
        case 'knn'
            k= learnerPar;
            % step 2-a:  fit model on training data using wieghts
            models{m} = fitcknn(datatrain_bs',dtrainbs,'NumNeighbors',k);
            % test traind madel(ith model) using all train data
            output = predict(models{m},datatrain')';
        case 'tree'
            Ns= learnerPar;
            % step 2-a:  fit model on training data using wieghts
            models{m} = fitctree(datatrain_bs',dtrainbs,'MaxNumSplit',Ns);
            % test traind madel(ith model) using all train data
            output = predict(models{m},datatrain')';
            
        otherwise
            error('not defined correct name for weak learner!')
    end
    % step 2-b: calculate weighted errer
    I= (output~=dtrain);
    erm= sum(D.* I) / sum(D);
    % step 2-d: calculate Alpha
    %     am(m)= log((1-erm) / erm);
    am(m)= 0.5*log((1-erm) / erm);
    % step 2-d: update weights
    D=D.* exp(am(m)* I);
    % step 2-e: renormalize weights
    D= D/ sum(D);
end
%%
mdl.models= models;
mdl.weaklearner= WL;
mdl.alpha= am;
mdl.NumModels= M;
mdl.userlabel=userlabel;
end

