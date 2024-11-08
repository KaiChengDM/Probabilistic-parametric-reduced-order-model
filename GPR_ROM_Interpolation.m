function [MSE,MSE1,Var,Var1,model] = GPR_ROM_Interpolation(x_test,Snapshots,Mu_t,Var_t,U_r,x_train)

%% Validate the accuracy of the pROM 
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
* MSE   : Prediction error
* MSE1  : Prediction error1
* Var   : Prediction standard deviation
* Var1  : Prediction standard deviation1
* model : Interpolation model
%}

%% Compute the relative error and standard deviation

model = Interpolation_model(x_train,Mu_t,Var_t);

[r, N_t] = size(Mu_t{1});   N = size(Mu_t,2);
 
 N1 = size(x_test,1);

 ub_input =  model{1}.ub_input;
 lb_input =  model{1}.lb_input;

 x_pre = (x_test - repmat(lb_input,N1,1))./(repmat(ub_input,N1,1)-repmat(lb_input,N1,1));

for i = 1: N1

     X_full = Snapshots(x_test(i,:));
    
     for k = 1:r
    
         std_y = model{k}.std_y;

         [weight Con_var] = Kriging_weight(x_pre(i,:),model{k});

         Mu_pred(k,:) = model{k}.mu_y; Var_pred(k,:) =  Con_var.*std_y.^2;
   
         for j = 1:N 
             Mu_pred(k,:)  = Mu_pred(k,:)  + weight(j)*(Mu_t{j}(k,:)- model{k}.mu_y);
             Var_pred(k,:) = Var_pred(k,:) + weight(j)^2*Var_t{j}(k,:);
         end

     end

         Var1(i) =  norm(sqrt(Var_pred),'fro')/norm(X_full,'fro');
         
         for j =1 :N_t
            Vart(j) = norm(sqrt(Var_pred(:,j)))/norm(X_full(:,j));
         end

         Var(i) = mean(Vart);

         Mu_full = U_r*Mu_pred;

         MSE1(i) =  norm(Mu_full - X_full,'fro')/norm(X_full,'fro');
 
         for j =1:N_t
            MSE_t(j) = norm(Mu_full(:,j) - X_full(:,j))/norm(X_full(:,j));
         end

        MSE(i) = mean(MSE_t);

 end

end