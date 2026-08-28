function [group] = myNBclassify(MDL,datatest)
% Manual implementation of Naive Bayes classifier 

muP=MDL.mu;
sigmaP=MDL.sigma;
pw=MDL.pw;
labels=MDL.Y;
for i=1:size(datatest,2)
    xi= datatest(:,i);
    for j= 1:size(muP,2)
        L(j) =  mypdf(xi,muP(:,j),sigmaP(:,:,j));
        ps(j) = L(j) * pw(j);
    end
    [~,ind] = max(ps);
    group(i)= labels(ind);
end
end

function [f]= mypdf(xi,muj,sigmaj)
f= (1/sqrt( (2*pi)^length(xi) *det(sigmaj))) * ...
        exp( -0.5 * (xi-muj)'*inv(sigmaj)*(xi-muj));
end