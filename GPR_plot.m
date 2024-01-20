
clc;  clear;

n=1;
 
g = @(x)exp(-x(:,1))+sin(2.*x(:,1))+cos(1.*x(:,1))+0.2.*x(:,1)+4;

Pd{1} = @(x)-exp(-x(:,1)) +2.* cos(2.*x(:,1))-1.*sin(1.*x(:,1))+0.2;

%% Sampling

 sig = ones(1,n); 
 mu  = zeros(1,n);
 lb  = 0.*ones(1,n);
 ub  = 5.*ones(1,n); 
 N   = 5;  
 N1  = 1000;
 
 pp = sobolset(n,'Skip',10); u=net(pp,N);  

% u = normcdf(lhsnorm(mu,diag(sig.^2),N));
 u1 = normcdf(lhsnorm(mu,diag(sig.^2),N1));

 for i = 1:n
    x(:,i) = u(1:N,i)*(ub(i)-lb(i))+lb(i);
    xtest(:,i) = u1(1:N1,i)*(ub(i)-lb(i))+lb(i);
 end

y = g(x);  y1 = g(xtest); 
 
 for i = 1:N
   Par =[];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,:));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
 end

 %% Kriging

hyperpar.theta = 0.1.*ones(1,n); 
hyperpar.lb    = 5*10^-4.*ones(1,n);
hyperpar.ub    = 5*ones(1,n);
% hyperpar.corr_fun = 'corrbiquadspline';
hyperpar.corr_fun      = 'corrgaussian';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts   = 10;

inputpar.x = x;
inputpar.y = y;

t1=clock;
  Kriging_Model = Kriging_fit(inputpar,hyperpar);
t2=clock;

Time1 = etime(t2,t1)
[Mean Variance] = Kriging_predictor(xtest,Kriging_Model);
MSE1  = mean((Mean-y1).^2)/var(y1)

%% plot

ii = lb:0.02:ub;

[Mean Var] = Kriging_predictor(ii',Kriging_Model);

CI = [Mean-1.96*sqrt(Var) Mean+1.96*sqrt(Var)];

plot(ii,g(ii'),'b-',"LineWidth",2); hold on

plot(ii,Mean,'m-',"LineWidth",2); hold on

plot(x,y,'ro',"LineWidth",2); hold on


for i = 1:length(ii)
   jj = ii(i)*ones(1,500);
   yy = CI(i,1):(CI(i,2)-CI(i,1))/499:CI(i,2);
   plot(jj,yy,'g-',"LineWidth",2);  hold on
   % alpha(.1)
end

plot(ii,g(ii'),'b-',"LineWidth",2); hold on

plot(ii,Mean,'m-',"LineWidth",2); hold on

plot(x,y,'ro',"LineWidth",2); hold on

plot(ii,CI(:,1),'g-',"LineWidth",2); hold on
plot(ii,CI(:,2),'g-',"LineWidth",2); hold on

legend('True function','GPR mean','Sample points','95% confidence interval')
xlabel('x')
ylabel('y')
