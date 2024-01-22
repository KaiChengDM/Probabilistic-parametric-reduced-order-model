function [Mu_full,Var_full,X_full] = GPR_ROM_prediction(x_test,Snapshots,Mu_t,Var_t,U_r,x_train)

%% Prediction of FOM for an untried parameter 
%{
Created by: Kai Cheng (kai.cheng@tum.de)
Based on: "ADAPTIVE DATA-DRIVEN PROBABILISTIC REDUCED-ORDER
MODELS FOR PARAMETERIZED DYNAMICAL SYSTEMS", submitted to SIAM journal on Scientific Computing
---------------------------------------------------------------------------
Input:
* Snapshots : Function for collecting snapshots
* x_test : Testing  parameter set
* Mu_t   : Mean of time sequence for training parameter set
* Var_t  : Variance of time sequence for training parameter set
* U_r    : Global basis
* x_train: Training parameter set
---------------------------------------------------------------------------
Output:
* Mu_full   : Prediction mean of the full order solution
* Var_full  : Prediction variance of the full order solution
* X_full    : True full order solution
%}

%% Prediction of FOM for an untried parameter 

model = Interpolation_model(x_train,Mu_t,Var_t);

N1 = size(x_test,1); [r, N_t] = size(Mu_t{1});   N = size(Mu_t,2);

ub_input =  model{1}.ub_input;
lb_input =  model{1}.lb_input;

x_pre = (x_test - repmat(lb_input,N1,1))./(repmat(ub_input,N1,1)-repmat(lb_input,N1,1));

for i = 1: N1

     tic
       X_full{i} = Snapshots(x_test(i,:));
     toc

     for k = 1:r
    
         std_y = model{k}.std_y;

         [weight Con_var] = Kriging_weight(x_pre(i,:),model{k});

         Mu_pred(k,:)  = model{k}.mu_y; 
         Var_pred(k,:) =  Con_var.*std_y.^2;
   
         for j = 1:N 
             Mu_pred(k,:)  = Mu_pred(k,:)  + weight(j)*(Mu_t{j}(k,:)- model{k}.mu_y);
             Var_pred(k,:) = Var_pred(k,:) + weight(j)^2*Var_t{j}(k,:);
         end

     end

         Mu_full{i} = U_r*Mu_pred;

         for j = 1:N_t
             Var_full{i}(:,j) = diag(U_r*diag(Var_pred(:,j))*U_r');
         end

 end


end