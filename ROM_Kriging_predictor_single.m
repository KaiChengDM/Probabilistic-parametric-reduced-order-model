function [X_mean,X_Var] = ROM_Kriging_predictor_single(Xtest,Kriging_Model,m)

% Prediction of state with Kriging
 
 output   = Kriging_Model{1}.Yr;
 U_r      = Kriging_Model{1}.basis;
 ub_input = Kriging_Model{1}.ub_input;

 X_r_test = U_r'*Xtest;

 r = size(output,2);

 u_test = X_r_test';    N = size(u_test,1); 
     
 u_test  = u_test./repmat(ub_input,N,1);

for j = 1:m

     for i = 1: r
         [Mean(j,i), Var(j,i)] = Kriging_predictor_single(u_test(j,:),Kriging_Model{i}); % assign same weight for each component of y_r
     end

     Mean(j,:) = Mean(j,:).*ub_input;
     Var(j,:)  = Var(j,:).*ub_input.^2;

end

  X_mean = U_r*Mean'; 
  X_Var  = U_r.^2*Var';

end