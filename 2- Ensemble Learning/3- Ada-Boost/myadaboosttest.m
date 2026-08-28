function [output] = myadaboosttest(mdl,datatest)
%  Algorithm       : AdaBoost (Adaptive Boosting) – from scratch
%  
%  Description     : Manual implementation of AdaBoost algorithm 
%                    (weight updating, weak learners combination) on the Iris dataset.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.

models= mdl.models;

WL=mdl.weaklearner;
alpha=mdl.alpha;
numModels=mdl.NumModels;
userlabel=mdl.userlabel;
% test trained classifiers using test data
for m=1:numModels
    switch WL
        case 'slp'
            P(m,:)= mySLPtest(models{m},datatest);
        otherwise
            P(m,:)= predict(models{m},datatest')';
    end
end
[output] = mywieghtedVoting(P,userlabel,alpha);

end

function [output] = mywieghtedVoting(P,userlabel,am)
NumClass= numel(userlabel);
for i=1:NumClass
    tp= (P==userlabel(i));
    reptA= repmat(am',1,size(P,2));
    tp= tp.*reptA;
    num(i,:)= sum(tp,1);
end
[~,indx]= max(num);
y= indx;
for i=1:NumClass
    ind= find(y==i);
    output(ind)= userlabel(i);
end
end