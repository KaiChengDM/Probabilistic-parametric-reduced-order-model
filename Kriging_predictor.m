function [pred_mu, pred_var] = Kriging_predictor(x_pre,model)  

% Kriging model predictor

%% Preparation

corr_fun = model.corr_fun; 

weight = model.weight; 

y = model.orig_output; 
  
switch corr_fun             
       case 'corrgaussian'
         corrvector = Gaussian_corrvector(x_pre,model,'off'); % Correlation vector 
       case 'corrspline'
         corrvector = Spline_corrvector(x_pre,model,'off'); % Correlation vector
       case 'corrbiquadspline'
        corrvector = Biquadspline_corrvector(x_pre,model,'off'); % Reduced correlation matrix
 end  

 %% Prediction

 u = model.tran_input; 

 corrvector = (1-weight)*corrvector + weight*x_pre*u';  % Mixed kernel 

 upper_mat = model.upper_mat;

 sigma2 = model.sigma2;

 mu= corrvector*(upper_mat\(upper_mat'\y));    % Kriging prediction mean
 
 rt = upper_mat'\corrvector'; 
 
 var =  sigma2*((1-weight)+weight*x_pre*x_pre'- sum(rt.^2) )';
 
%  variance = ((1-weight)+weight*u_pre*u_pre'- sum(rt.^2) )';

 pred_mu  = mu;               % Original prediction mean
 pred_var = var;              % Original prediction variance


end
