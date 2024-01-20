function [Var Delta] = Adaptive_GPR_ROM(x_train,x_test,Mu_t,Var_t)

% Compute the bias and variance (all latent states shate the same hyper-parameter)

%% training sample set

 model = Interpolation_model(x_train,Mu_t,Var_t);

 [r, N_t] = size(Mu_t{1});   N = size(Mu_t,2);
 
 N1 = size(x_test,1);

 ub_input =  model{1}.ub_input;
 lb_input =  model{1}.lb_input;

 x_pre = (x_test - repmat(lb_input,N1,1))./(repmat(ub_input,N1,1)-repmat(lb_input,N1,1));

 for i = 1: N1 
    
     for k = 1:r
         
         std_y = model{k}.std_y;

         [weight Con_var] = Kriging_weight(x_pre(i,:),model{k});

         Mu_pred(k,:) = model{k}.mu_y; Var_pred(k,:) = Con_var.*std_y.^2;
   
         for j = 1:N    
             Mu_pred(k,:)  = Mu_pred(k,:)  + weight(j)*(Mu_t{j}(k,:)- model{k}.mu_y);
             Var_pred(k,:) = Var_pred(k,:) + weight(j)^2*Var_t{j}(k,:);
         end
         
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