function [ Mu_t,Var_t,U_r,X_test, ROM_Kriging ] = GPR_ROM(parameter,snapshot,m,threshold)

%% Construct ROM based on GPR
%{
Created by: Kai Cheng (kai.cheng@tum.de)
Based on: "ADAPTIVE DATA-DRIVEN PROBABILISTIC REDUCED-ORDER
MODELS FOR PARAMETERIZED DYNAMICAL SYSTEMS", submitted to SIAM journal on Scientific Computing
---------------------------------------------------------------------------
Input:
* parameter: Training parameter set
* snapshot : Function for collecting snapshots
* m        : Training data sequence length
* threshold: SVD truncation threshold
---------------------------------------------------------------------------
Output:
* Mu_t       : Prediction mean for all sampled parameter value 
* Var_t      : Prediction variance for all sampled parameter value 
* U_r        : Global basis
* X_test     : Test snapshots
* ROM_Kriging: Trained Kriging model for all sampled parameter value 
%}

%% training sample set

N = size(parameter,1); 

X_t  = [];  X_test = [];

for kk = 1:N

    a = parameter(kk,:); 
 
    [X1 X2 X_test{kk}] = snapshot(a,m);

    X_train = [X1 X2(:,end)];  % total snapshots a sampled parameter
 
    X_t = [X_t X_train];       % total snapshots for all sampled parameters

end

%% find global basis

[U, S, V] = svd(X_t, 'econ');

Sigma = diag(S);

for i = 1 :length(Sigma)
    % energy(i) = sum(Sigma(1:i).^2)./sum(Sigma.^2);
    energy(i) = sum(Sigma(1:i))./sum(Sigma); 
    if energy(i)  > threshold
        r = i;
        break;
    end
end

U_r = U(:, 1:r);     % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);

X_r = U_r'*X_t;      % Low-dimensional time-dependent coefficientss

% plot(1:m+1,X_train(40,:))

%%  Kriging

mm = size(X_test{1},2);

hyperpar.corr_fun    = 'corrgaussian';
% hyperpar.corr_fun    = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts   = 10;

for kk = 1 : N

    X_train = X_r(:,(m+1)*(kk-1)+1:kk*(m+1));
   
    error = 1;

    while error > 5*10^-4

         ROM_Kriging = ROM_Kriging_train(X_train,hyperpar); 
     
         % Recover training data
         Xtest          = X_train;
         Mu_r{kk}(:,1)  = Xtest(:,1); 
         Var_r{kk}(:,1) = zeros(r,1);

         for i = 1:m
             [Mu_r{kk}(:,i+1),Var_r{kk}(:,i+1)] = ROM_Kriging_predictor(Xtest(:,i),ROM_Kriging); 
             rec_err(i) = norm(Mu_r{kk}(:,i+1) - X_train(:,i+1))./norm(X_train(:,i+1));  % reconstruction error
         end

         % future state prediction

         Mu_r1{kk} = [];  Var_r1{kk} = [];
         Xtest = X_train(:,end);

         for i = 1:mm  % auto-regression
             [Mu_r1{kk}(:,i),Var_r1{kk}(:,i)] = ROM_Kriging_predictor(Xtest,ROM_Kriging); 
             Xtest = Mu_r1{kk}(:,i);   
         end

         Mu_t{kk}  = [Mu_r{kk}   Mu_r1{kk}];  
         Var_t{kk} = [Var_r{kk} Var_r1{kk}];

         error = norm(sqrt(Var_t{kk}),'fro')/norm(Mu_t{kk},'fro');

         if error > 5*10^-4
            error
            fprintf("Surrogate model may not be accurate enough ! Try to re-train \n");
         end

      end

   end

end