function [X_mean,X_Variance] = POD_Kriging_predictor(Xtest,Kriging_Model)

% Prediction of future state with ROM_Kriging

 U_r = Kriging_Model{1}.basis;

 r = Kriging_Model{1}.rank;

 lb_input    = Kriging_Model{1}.lb_input;
 ub_input    = Kriging_Model{1}.ub_input;
 mean_output = Kriging_Model{1}.mean_output;
 std_output  = Kriging_Model{1}.std_output;

 N = size(Xtest,1); 
     
 u_test  = (Xtest - repmat(lb_input,N,1))./(repmat(ub_input,N,1)-repmat(lb_input,N,1));

 for i = 1: r

     [Mean(:,i) Variance(:,i)] = Kriging_predictor_single(u_test,Kriging_Model{i}); % assign same weight for each component of y_r
   
     Mean(:,i) = Mean(:,i)*std_output(i)+ mean_output(i);

     Variance(:,i) = Variance(:,i)*std_output(i)^2;
    
%      Mean(:,i) = Mean(:,i)*(ub_input-lb_input) + lb_input;
% 
%      Variance(:,i) = Variance(:,i)*(ub_input-lb_input)^2;

 end

  X_mean = U_r*Mean';

  X_Variance = U_r.^2*Variance';

end