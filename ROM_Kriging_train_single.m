function ROM_Kriging = ROM_Kriging_train_single(X,threshold,hyperpar)

% Training reduced order model with Kriging with single kernel

[U, S, V] = svd(X, 'econ');

Sigma = diag(S);

for i = 1 :length(Sigma)
    energy(i) = sum(Sigma(1:i))./sum(Sigma);
    if energy(i)  > threshold
        r = i;
        break;
    end
end

U_r = U(:, 1:r);                % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);

X_r = U_r'*X;  % Input 

n = r;  m = size(X_r,2);

u_input  = X_r(:,1:m-1)';   
u_output = X_r(:,2:m)';

N = length(u_input); 

ub_input = max(abs(u_input));

u_input  = u_input./repmat(ub_input,N,1);
u_output = u_output./repmat(ub_input,N,1);

x = u_input; 

%% All hyperparameter share the same value 

hyperpar.theta = 0.1; 
hyperpar.lb    = 10^-7;
hyperpar.ub    = 20;

for k = 1:n

  y = u_output(:,k);  

  inputpar.x = x;
  inputpar.y = y;

  t1 = clock;
    ROM_Kriging{k} = Kriging_fit_single(inputpar,hyperpar);  % training Kriging model 
  t2 = clock;

  ROM_Kriging{k}.Yr = u_output;  % multiple output

end

 ROM_Kriging{1}.basis = U_r; 
 ROM_Kriging{1}.ub_input   = ub_input;

end