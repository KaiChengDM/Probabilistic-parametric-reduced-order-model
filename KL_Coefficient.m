function Y = KL_Coefficient(u,meant,sigmat,L)

% K-L expansion of Stochastic Process 

% meant = 1 ; sigmat = 0.2;  L = 0.5;

 rou = @(x1,x2)exp(-norm(x1-x2)^2/L^2);
 
 x = 0:0.01:1; 

 l = length(x);

 for i = 1:l
     for j = 1:l
       Rou(i,j) = rou(x(i),x(j));
     end
 end

 [v,d] = eig(Rou);        %eigenvalues and eigenvectors of correlation matrix

 eigenvalues =diag(d);

 [eigenvalues,vv]=sort(eigenvalues,'descend');   %sort the eigenvalues and eigenvectors

 for i = 1 :length(eigenvalues)
    energy(i) = sum(eigenvalues(1:i))./sum(eigenvalues);
    if energy(i)  > 0.999
        r = i;
        break;
    end
 end

 for i = 1:r
     V(:,i) = v(:,vv(i));     %eigenvectors
 end

 N1 = l;

 Y = zeros(1,N1);

 for j = 1:N1
   for i = 1:r
      Y(j) = Y(j)+(u(i)./(sqrt(eigenvalues(i)))).*(V(:,i)'*Rou(j,:)');    %    The expansion of the stochastic process
   end
 end

 Y = meant + sigmat.*Y;

end