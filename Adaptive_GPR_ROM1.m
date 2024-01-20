function [Var Delta] = Adaptive_GPR_ROM1(x_train,x_test,Mu_t,Var_t)

% Compute the bias and variance (all latent states shate the same hyper-parameter)

%% training sample set

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

 [weight Con_var] = Kriging_weight(x_pre,model);

 for i = 1: N1 
     
     Var_pred = Con_var(i).*std_y.^2; Mu_pred = mu_y;
   
     for j = 1:N 
         Mu_pred  = Mu_pred  + weight(i,j)*(Mu_t{j}-mu_y);
         Var_pred = Var_pred + weight(i,j)^2*Var_t{j};
     end

     Var(i) = norm(sqrt(Var_pred),'fro')^2;       % variance

     if (length(x_test(i,:))==1)
          [value ind] = min(abs(x_test(i,:) - x_train));
     else
          [value ind] = min(vecnorm((x_test(i,:) - x_train)'));
     end

     Delta(i) = norm(Mu_pred - Mu_t{ind},'fro')^2; % bias

 end

end