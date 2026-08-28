function [mdl] = mysvmtrain(Xtrain,Ytrain,kernel,kernelPar,Constrain,flag)
%  Algorithm       : Binary Support Vector Machine (SVM) - from scratch
%  
%  Description     : Manual implementation of Binary SVM (Hard-Margin and 
%                    Soft-Margin) with Linear, RBF, and 
%                    Polynomial kernels, including the dual formulation, quadratic 
%                    programming, and support vector identification.
%                    Part of the comprehensive Pattern Recognition & 
%                    Machine Learning course.
type_label= unique(Ytrain);
if numel(type_label)> 2
    error(' number labels is greater than 2!')
end
ind1= find(Ytrain== type_label(1) );
ind2= find(Ytrain== type_label(2) );
group(ind1)= 1;
group(ind2)= -1;
%% ******************** train classifier ***********************
X= Xtrain; % train input
Y= group; % train output
N= size(Xtrain,2);
%% step 1: cost function
% ****** define H ******
% nonlinear mapping
C=Constrain;
[ker] = mykernelfunction(X,X,kernel,kernelPar);
H= (Y'*Y).* (ker);
% ***** define f ******
f= -ones(N,1); 
%% step 2: define inequality constraints
A= [];
b= []; 
%% step 3: define equality constraints
Aeq= Y;
beq= 0; 
%% step 4: define boundaries
lb= zeros(N,1); 
ub= ones(N,1)*C;
% initialize optimization variable
x0= [];
%% step 5: define quadratic programming solver options
options= optimoptions('quadprog', 'MaxIteration',500,'Display','off');
%% step 6: solve quadratic programming problem
[alpha,~,exitflag,~,lambda]= quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
%% step 7: find support vectors
percision= 0.001;
ind_sv= find(alpha>percision &  alpha< C-percision );
sv= X(:,ind_sv);
%% step 9: calculate b
bo= lambda.eqlin;
%%
mdl.Ytrain= Y;
mdl.Xtrain= Xtrain;
mdl.alpha= alpha;
mdl.b=bo;
mdl.SupportVectors=sv;
mdl.type_label=type_label;
mdl.kernel= kernel;
mdl.kernelparameter= kernelPar;

if (flag==1)
    colr=rand(3,1);
    plot(sv(1,:),sv(2,:),'o','linewidth',2,'markersize',8,...
        'markerEdgecolor',colr,'markerfacecolor',colr)

    wo= ((alpha.*Y')' * X')';
    %% decision boundary
    x1=-20:20;
    x2= (-wo(1)/wo(2))*x1 - (bo/wo(2));
    plot(x1,x2,'Color',colr,'linewidth',3)
end

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





