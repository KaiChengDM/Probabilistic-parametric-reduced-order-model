function Krigingmodel = Interpolation_model1(x_train,Mu_t,Var_t)

% Interpolation of latent states based on GPR - all components share the same hyper-parameter

%% training sample set

n = size(x_train,2);  N = size(Mu_t,2);  

hyperpar.corr_fun      = 'corrgaussian';
% hyperpar.corr_fun    = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts   = 10;
hyperpar.theta =  0.1.*ones(1,n); 
hyperpar.lb    =  10^-4.*ones(1,n);
hyperpar.ub    =  10*ones(1,n);

p_lb  = max(x_train);  p_ub = min(x_train);

x_train = (x_train - repmat(p_lb,N,1))./(repmat(p_ub,N,1)-repmat(p_lb,N,1)); % normalized input data

[r , m]= size(Mu_t{1});

for i = 1:N
    y(i,:)   = Mu_t{i}(1,:);
    var(i,:) = Var_t{i}(1,:);
end
  
mu_y  = mean(y);
std_y = std(y);

std_y(find(std_y ==0))=1;

y = (y- mu_y)./std_y; var = var./std_y.^2;     % normalized output data
   
inputpar.x     = x_train;
inputpar.y     = y;
inputpar.var   = var;
inputpar.std_y = std_y;
inputpar.nt    = m;
inputpar.mu_y  = mu_y;

Krigingmodel = Kriging_hyperparameter(inputpar,hyperpar);  % tune Kriging hyper-parameter

Krigingmodel.ub_input = p_ub;
Krigingmodel.lb_input = p_lb;

end