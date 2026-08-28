function [mdl] = mySVRtrain(Xtrain,Ytrain,kernel,kernelPar,C,epsilon)
%  Algorithm       : Support Vector Regression (SVR) with Linear, RBF, 
%                    and Polynomial kernels – from scratch
%  
%  Description     : Manual implementation of Support Vector Regression 
%                    (SVR) using Linear, RBF, and Polynomial kernels.
%                    Applied to the UCI Air Quality dataset.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.

%% ******************** train Model ***********************
X= Xtrain; % train input
Y= Ytrain; % train output
N= size(Xtrain,2);

%% step 1: cost function
% ****** determine epsilon & C
% C=Constraint;
% epsilon=0.1;
% % ****** define H ******
% kernel='rbf';
% kernelPar=1;
D = mykernelfunction(X,X,kernel,kernelPar);
H= [D,-D;-D,D];
% % ***** define f ******
f= [epsilon-Y,epsilon+Y]';
%% step 2: define inequality constraints
A= [];
b= []; 
% %% step 3: define equality constraints
Aeq= [ones(1,N),-1*ones(1,N)];
beq= 0; 
% %% step 4: define boundaries
lb= [zeros(1,N),zeros(1,N)]'; 
ub= [C*ones(1,N),C*ones(1,N)]'; 
% initialize optimization variable
x0= [];
%% step 5: define quadratic programming solver options
options= optimoptions('quadprog','MaxIteration',500,'Display','off');
%% step 6: solve quadratic programming problem
[beta,fval,exitflag,outputs,lambda]= quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
%% step 7: calculate (a-a*)
alpha= beta(1:N);
alpha_s= beta(N+1:end);
alphan= (alpha-alpha_s);
%% step 8: calculate w
% wo= (alphan)'*X';
%% step 9: calculate b
bo=lambda.eqlin ;

mdl.Xtrain=Xtrain;
mdl.kernel=kernel;
mdl.kernelparameter=kernelPar;
mdl.bo=bo;
mdl.alphan=alphan;
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
