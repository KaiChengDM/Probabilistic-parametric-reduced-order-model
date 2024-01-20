function [MSE,MSE1,Var,Var1,model] = GPR_ROM_Interpolation1(x_test,Snapshots,Mu_t,Var_t,U_r,x_train)

% Interpolation of ROM based on GPR (all latent states share the same hyper-parameter with the first one)


model = Interpolation_model1(x_train,Mu_t,Var_t);

[r, N_t] = size(Mu_t{1});   N = size(Mu_t,2);
 
 N1 = size(x_test,1);

 ub_input =  model.ub_input;
 lb_input =  model.lb_input;

 x_pre = (x_test - repmat(lb_input,N1,1))./(repmat(ub_input,N1,1)-repmat(lb_input,N1,1));

for k = 1: r 
    for i = 1:N
        y(i,:)   = Mu_t{i}(k,:);
        var(i,:) = Var_t{i}(k,:);
    end
    mu_y(k,:)  = mean(y);
    std_y(k,:) = std(y);
end


for i = 1: N1

     X_full = Snapshots(x_test(i,:));
    
     [weight Con_var] = Kriging_weight(x_pre(i,:),model);

     Var_pred = Con_var.*std_y.^2; Mu_pred = mu_y;
   
     for j = 1:N 
         Mu_pred  = Mu_pred  + weight(j)*(Mu_t{j}-mu_y);
         Var_pred = Var_pred + weight(j)^2*Var_t{j};
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