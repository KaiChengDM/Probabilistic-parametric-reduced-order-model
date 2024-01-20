function [X_mean,X_Variance] = ROM_Kriging_predictor_single(Xtest,Kriging_Model,m)

% Prediction of future state with ROM_Kriging
 
 output = Kriging_Model{1}.Yr;
 U_r  = Kriging_Model{1}.basis;
 ub_input = Kriging_Model{1}.ub_input;

 X_r_test = U_r'*Xtest;

 r = size(output,2);

 u_test = X_r_test';    N = size(u_test,1); 
     
 u_test  = u_test./repmat(ub_input,N,1);

for j = 1:m

     for i = 1: r
         [Mean(j,i) Variance(j,i)] = Kriging_predictor_single(u_test(j,:),Kriging_Model{i}); % assign same weight for each component of y_r
     end

%      u_test = Mean(j,:);
    
     Mean(j,:) = Mean(j,:).*ub_input;
     Variance(j,:) = Variance(j,:).*ub_input.^2;

end

  X_mean = U_r*Mean'; 

  X_Variance = U_r.^2*Variance';

end