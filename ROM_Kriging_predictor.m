function [X_mu,X_Var] = ROM_Kriging_predictor(Xtest,Kriging_Model)

% Prediction of state with Kriging

 ub_data = Kriging_Model{1}.ub_data;

 r = size(Kriging_Model,2);

 u_test  = Xtest./ub_data';  

 for i = 1: r

      [Mu(:,i), Var(:,i)] = Kriging_predictor(u_test',Kriging_Model{i}); 
    
      Mu(:,i)  = Mu(:,i)*ub_data(i);
      Var(:,i) = Var(:,i)*ub_data(i)^2;

 end

  X_mu  = Mu;
  X_Var = Var;

end