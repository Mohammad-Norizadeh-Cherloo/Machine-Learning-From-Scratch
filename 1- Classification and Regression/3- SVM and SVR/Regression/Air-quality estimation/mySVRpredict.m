function [output] = mySVRpredict(mdl,Xtest)
%  Algorithm       : Support Vector Regression (SVR) with Linear, RBF, 
%                    and Polynomial kernels – from scratch
%  
%  Description     : Manual implementation of Support Vector Regression 
%                    (SVR) using Linear, RBF, and Polynomial kernels.
%                    Applied to the UCI Air Quality dataset.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
Xtrain= mdl.Xtrain;
kernel=mdl.kernel;
kernelPar= mdl.kernelparameter;
bo= mdl.bo;
alphan= mdl.alphan;
[ker] = mykernelfunction(Xtrain,Xtest,kernel,kernelPar);
output= (alphan'*ker)+bo;
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