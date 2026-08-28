function [output] = mymultisvmclassifyOVO(mdl,Xtest)
%  Algorithm       : Multi-class SVM (One-vs-One) with Linear, RBF, and 
%                    Polynomial kernels – from scratch
%  
%  Description     : Manual implementation of Multi-class Support Vector 
%                    Machine using the One-vs-One (OvO) strategy.
%                    Includes Linear, RBF, and Polynomial kernels.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
userlabel= mdl.userlabel;
%%
svm1=mdl.svm1;
output1 = mysvmclassify(svm1,Xtest);

svm2=mdl.svm2;
output2 = mysvmclassify(svm2,Xtest);

svm3=mdl.svm3;
output3 = mysvmclassify(svm3,Xtest);
%%
temp=[output1;output2;output3];
for i= 1:numel(userlabel)
    num(i,:)= sum( temp== userlabel(i));
end
[mx,ind]= max(num);
output= zeros(1,size(Xtest,2));

for i= 1:numel(userlabel)
    index = find(ind==i);
   output(index)= userlabel(i);
end

end
