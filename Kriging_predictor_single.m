function [pred_mean pred_variance ] = Kriging_predictor_single(x_pre,model)  

% Kriging model predictor

%% Preparation

corr_fun = model.corr_fun;  

u_pre = x_pre;

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

 upper_mat = model.upper_mat;

 f =  model.f ;   

 beta0 = model.beta0;   sigma2 = model.sigma2;

 mean = beta0+corrvector*(upper_mat\(upper_mat'\(y-f*beta0)));    % Kriging prediction mean

 rt = upper_mat'\corrvector'; 
  
  variance =  sigma2*(1- sum(rt.^2))';

  pred_mean     = mean;
  pred_variance = variance;

%  output_moment = model.output_moment; 
%  pred_mean     = mean.*output_moment(2)+output_moment(1);    % Original prediction mean
%  pred_variance = variance.*output_moment(2).^2;              % Original prediction variance

end
