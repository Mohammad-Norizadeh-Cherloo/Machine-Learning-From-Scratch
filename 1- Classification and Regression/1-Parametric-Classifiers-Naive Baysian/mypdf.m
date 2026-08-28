function [f]= mypdf(xi,muj,sigmaj)
f= (1/sqrt( (2*pi)^length(xi) *det(sigmaj))) * ...
        exp( -0.5 * (xi-muj)'*inv(sigmaj)*(xi-muj));
end