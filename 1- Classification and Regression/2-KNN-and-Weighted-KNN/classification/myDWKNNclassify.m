function [output] = myDWKNNclassify(MDL,datatest)
% Weighted KNN Classifier (from scratch)
datatrain=MDL.datatrain;
dtrain=MDL.dtrain;
%% step 1: determine Num neighbors (k)
k=MDL.NumNeighbors;
Distance=MDL.Distance;
Y=MDL.Y;
switch (Distance)
    case 'Euclidean'
        for i=1:size(datatest,2)
            A= datatest(:,i);
            %% step 2: calculate distacne between xtest and train data
            for j=1:size(datatrain,2)
                B= datatrain(:,j);
                dis(j)= sqrt(sum((A-B).^2));
            end
            %% step 3: sort distance
            [dis,ind]= sort(dis);
            %% step 4: find k nearest neigbors from train data to xtest
            NN_Labels= dtrain(ind(1:k));
            D= dis(1:k);
            numClass= numel(Y);
            
            if D(k)==D(1)
                w= ones(1,k);
            elseif D(k)~=D(1)
                w= ((D(k)-D) /(D(k)-D(1))).* ((D(k)+D(1)) ./(D(k)+D)); 
            end
            %% step 5: voting (classify xtest)
            for n=1:numClass
                N(n) = sum(w.*(NN_Labels==Y(n)));
            end
            [mx,indx]= max(N);
            output(i)=Y(indx);
            
        end
    case 'Cityblock'
        for i=1:size(datatest,2)
            A= datatest(:,i);
            %% step 2: calculate distacne between xtest and train data
            for j=1:size(datatrain,2)
                B= datatrain(:,j);
                dis(j)= sum(abs(A-B));
            end
            %% step 3: sort distance
            [dis,ind]= sort(dis);
            %% step 4: find k nearest neigbors from train data to xtest
            NN_Labels= dtrain(ind(1:k));
            D= dis(1:k);
            numClass= numel(Y);
            if D(k)==D(1)
                w= ones(1,k);
            elseif D(k)~=D(1)
                w= ((D(k)-D) /(D(k)-D(1))).* ((D(k)+D(1)) ./(D(k)+D)); 
            end
            
            %% step 5: voting (classify xtest)
            for n=1:numClass
                N(n) = sum(w.*(NN_Labels==Y(n)));
            end
            [mx,indx]= max(N);
            output(i)=Y(indx);
            
        end
    case 'Chebychev'
        for i=1:size(datatest,2)
            A= datatest(:,i);
            %% step 2: calculate distacne between xtest and train data
            for j=1:size(datatrain,2)
                B= datatrain(:,j);
                dis(j)= max(abs(A-B));
            end
            %% step 3: sort distance
            [dis,ind]= sort(dis);
            %% step 4: find k nearest neigbors from train data to xtest
            NN_Labels= dtrain(ind(1:k));
            D= dis(1:k);
            numClass= numel(Y);
            if D(k)==D(1)
                w= ones(1,k);
            elseif D(k)~=D(1)
                w= ((D(k)-D) /(D(k)-D(1))).* ((D(k)+D(1)) ./(D(k)+D)); 
            end
            
            %% step 5: voting (classify xtest)
            for n=1:numClass
                N(n) = sum(w.*(NN_Labels==Y(n)));
            end
            [mx,indx]= max(N);
            output(i)=Y(indx);
            
        end
    case 'Minkowski'
        p=5;
        for i=1:size(datatest,2)
            A= datatest(:,i);
            %% step 2: calculate distacne between xtest and train data
            for j=1:size(datatrain,2)
                B= datatrain(:,j);
                dis(j)=  nthroot(sum(abs(A-B).^p),p);
            end
            %% step 3: sort distance
            [dis,ind]= sort(dis);
            %% step 4: find k nearest neigbors from train data to xtest
            NN_Labels= dtrain(ind(1:k));
            D= dis(1:k);
            numClass= numel(Y);
            if D(k)==D(1)
                w= ones(1,k);
            elseif D(k)~=D(1)
                w= ((D(k)-D) /(D(k)-D(1))).* ((D(k)+D(1)) ./(D(k)+D)); 
            end
            
            %% step 5: voting (classify xtest)
            for n=1:numClass
                N(n) = sum(w.*(NN_Labels==Y(n)));
            end
            [mx,indx]= max(N);
            output(i)=Y(indx);
            
        end
    case 'Cosine'
        for i=1:size(datatest,2)
            A= datatest(:,i);
            %% step 2: calculate distacne between xtest and train data
            for j=1:size(datatrain,2)
                B= datatrain(:,j);
                dis(j)=  1- ((A'*B) / sqrt((A'*A)*(B'*B)));
            end
            %% step 3: sort distance
            [dis,ind]= sort(dis);
            %% step 4: find k nearest neigbors from train data to xtest
            NN_Labels= dtrain(ind(1:k));
            D= dis(1:k);
            numClass= numel(Y);
            if D(k)==D(1)
                w= ones(1,k);
            elseif D(k)~=D(1)
                w= ((D(k)-D) /(D(k)-D(1))).* ((D(k)+D(1)) ./(D(k)+D)); 
            end
            
            %% step 5: voting (classify xtest)
            for n=1:numClass
                N(n) = sum(w.*(NN_Labels==Y(n)));
            end
            [mx,indx]= max(N);
            output(i)=Y(indx);
            
        end
    case 'Correlation'
        for i=1:size(datatest,2)
            A= datatest(:,i);
            %% step 2: calculate distacne between xtest and train data
            for j=1:size(datatrain,2)
                B= datatrain(:,j);
                dis(j)= 1-corr(A,B);
            end
            %% step 3: sort distance
            [dis,ind]= sort(dis);
            %% step 4: find k nearest neigbors from train data to xtest
            NN_Labels= dtrain(ind(1:k));
            D= dis(1:k);
            numClass= numel(Y);
            if D(k)==D(1)
                w= ones(1,k);
            elseif D(k)~=D(1)
                w= ((D(k)-D) /(D(k)-D(1))).* ((D(k)+D(1)) ./(D(k)+D)); 
            end
            
            %% step 5: voting (classify xtest)
            for n=1:numClass
                N(n) = sum(w.*(NN_Labels==Y(n)));
            end
            [mx,indx]= max(N);
            output(i)=Y(indx);
            
        end
    otherwise
        error('not defined correct name for Distance')
end
end