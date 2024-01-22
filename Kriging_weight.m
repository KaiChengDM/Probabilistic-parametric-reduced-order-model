function [weight, var] = Kriging_weight(x_pre,model)
 
% compute the interpolation weight and variance
  
 corr_fun = model.corr_fun; 

 switch corr_fun             
       case 'corrgaussian'
         corrvector = Gaussian_corrvector(x_pre,model,'off'); % Correlation vector 
       case 'corrspline'
         corrvector = Spline_corrvector(x_pre,model,'off'); % Correlation vector
       case 'corrbiquadspline'
        corrvector = Biquadspline_corrvector(x_pre,model,'off'); % Reduced correlation matrix
 end 

 upper_mat = model.upper_mat; 

 weight = corrvector/upper_mat/upper_mat';
 
 rt = upper_mat'\corrvector'; 

 var =  (1- sum(rt.^2))';

end


