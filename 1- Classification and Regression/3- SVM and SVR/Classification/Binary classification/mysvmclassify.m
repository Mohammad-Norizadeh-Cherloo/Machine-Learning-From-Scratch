function [output] = mysvmclassify(mdl,Xtest)
%  Algorithm       : Binary Support Vector Machine (SVM) - from scratch
%  
%  Description     : Manual implementation of Binary SVM (Hard-Margin and 
%                    Soft-Margin) with Linear, RBF, and 
%                    Polynomial kernels, including the dual formulation, quadratic 
%                    programming, and support vector identification.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.

Y=mdl.Ytrain;
Xtrain=mdl.Xtrain;
alpha=mdl.alpha;
bo=mdl.b;
type_label=mdl.type_label;
kernel=mdl.kernel;
kernelPar=mdl.kernelparameter;
%%
[ker] = mykernelfunction(Xtrain,Xtest,kernel,kernelPar);
output_p= sign( (Y.*alpha') * ker +bo);
%%
ind1= find(output_p== 1 );
ind2= find(output_p== -1 );
output(ind1)= type_label(1);
output(ind2)= type_label(2);
end
function [ker] = mykernelfunction(Xi,Xj,kernel,par)
%% kernel: linear, rbf,polynomial
switch kernel
    case ('linear')
        ker= Xi'*Xj;
    case ('polynomial')
        d= par;
        ker= (Xi'*Xj + 1).^d;
    case('rbf')
        sigma= par;
        for i=1:size(Xi,2)
            xi= Xi(:,i);
            for j=1:size(Xj,2)
                xj= Xj(:,j);
                ker(i,j)= exp((-0.5/sigma^2)*norm(xi-xj)^2);
            end
        end
    otherwise
        error('not defined correct kernel(linear, rbf,polynomial)')
end
end
