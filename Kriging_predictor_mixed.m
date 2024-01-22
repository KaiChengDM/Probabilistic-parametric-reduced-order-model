function [pred_mean, pred_variance] = Kriging_predictor_mixed(x_pre,model)  

% Kriging model predictor

%% Preparation

corr_fun = model.corr_fun; 

u_pre = x_pre;

weight = model.weight; 

y = model.orig_output; 

switch corr_fun             
       case 'corrgaussian'
         corrvector = Gaussian_corrvector(u_pre,model,'off'); % Correlation vector 
       case 'corrspline'
         corrvector = Spline_corrvector(u_pre,model,'off'); % Correlation vector
       case 'corrbiquadspline'
        corrvector = Biquadspline_corrvector(u_pre,model,'off'); % Reduced correlation matrix
 end  

 %% Prediction

 u = model.tran_input; 

 corrvector = (1-weight)*corrvector + weight*u_pre*u';  % Mixed kernel 

 upper_mat = model.upper_mat;

 sigma2 = model.sigma2;

 mean = corrvector*(upper_mat\(upper_mat'\y));    % Kriging prediction mean
 
  rt = upper_mat'\corrvector'; 
 
  variance =  sigma2*((1-weight)+weight*u_pre*u_pre'- sum(rt.^2) )';
 
%  variance = ((1-weight)+weight*u_pre*u_pre'- sum(rt.^2) )';

 pred_mean     = mean;                  % Original prediction mean
 pred_variance = variance;              % Original prediction variance


end
