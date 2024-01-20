
clear; clc;  close all;

%% Initial sampling 

lb = 100; ub = 800;  % low bound and upper bound

dim = 1;  N = 3;     % parameter dimension and initial sample size 

par = lb:(ub-lb)/(N-1):ub; par = par'; 

m = 100; threshold = 0.99999;  % snapshots number and SVD truncation threshold 

%% Initial model

N1 = 300;  pp = sobolset(dim,'Skip',10); u1 = net(pp,N1); % candidate sample by sobol sequence

for i = 1:dim
    p_pool(:,i)= u1(:,i)*(ub(i)-lb(i))+lb(i);   % candidate sample pool
end

x_train = par;  
 
[Mu_t,Var_t,U_r, X_test, ROM_Kriging] = GPR_ROM(x_train,@Burger_snapshots,m,threshold); % train GPR model 

% Assess the accuracy of initial pROM

[MSE_test,MSE_test1,Var_test,Var_test1] = GPR_ROM_Interpolation(p_pool,@Burger_snapshots_test,Mu_t,Var_t,U_r,x_train);

MSE_m(1) = mean(MSE_test); MSE_m1(1) = mean(MSE_test1); Var_m(1) = mean(Var_test);

%% adaptive sampling stage

for i = 1:7
 
    [Var Delta] = Adaptive_GPR_ROM(x_train,p_pool,Mu_t,Var_t);  % compute bias and variance

    Var1 = Var./max(Var);  Delta1 = Delta./max(Delta);

    [value ind] = max(Var1 + Delta1);        % select new parameter 
   
    x_train = [x_train; p_pool(ind,:)];

    [Mu_t,Var_t,U_r,X_test, ROM_Kriging] = GPR_ROM(x_train,@Burger_snapshots,m,threshold); % update GPR model
    
    % assess the accuracy of pROM

    [MSE_test,MSE_test1, Var_test] = GPR_ROM_Interpolation(p_pool,@Burger_snapshots_test,Mu_t,Var_t,U_r,x_train);

    MSE_m(i+1) = mean(MSE_test); MSE_m1(i+1) = mean(MSE_test1); Var_m(i+1) = mean(Var_test);

end

% convergenve curve 

subplot(1,2,1)
plot(0:6, MSE_m,'-o','LineWidth',1.5); hold on 
ylabel('AMRE');
xlabel('Iterations');

subplot(1,2,2)
plot(0:6,Var_m,'-o','LineWidth',1.5); hold on 
ylabel('AMRStd');
xlabel('Iterations');

%% Prediction for different parameter

test = (100:0.5:800)';

N1 = length(test);

x_test = test;
  
[MSE_test,MSE_test1,Var_test,Var_test1,model] = GPR_ROM_Interpolation(x_test,@Burger_snapshots_test,Mu_t,Var_t,U_r,x_train);

N = 3;
figure
subplot(1,2,1)
plot(100+(1:N1)*0.5,MSE_test1,'-','LineWidth',1.5); hold on 
plot(x_train(1:N),zeros(1,N),'mo','LineWidth',1.5); hold on 
plot(x_train(N+1:end),zeros(1,6),'rsquare','LineWidth',1.5); hold on 
ylabel('MRE');
xlabel('\omega');
legend('','Initial samples','Enriched samples')

subplot(1,2,2)
plot(100+(1:N1)*0.5,Var_test,'-','LineWidth',1.5); hold on 
plot(x_train(1:N),zeros(1,N),'mo','LineWidth',1.5); hold on 
plot(x_train(N+1:end),zeros(1,7),'rsquare','LineWidth',1.5); hold on 
ylabel('MRStd');
legend('','Initial samples','Enriched samples')
% legend('Relative error','Predicted standard deviation','Sampled points')
xlabel('\omega');

%% single paramter prediction

x_test = 680;

[Mu_full,Var_full,X_full] = GPR_ROM_prediction(x_test,@Burger_snapshots_test,Mu_t,Var_t,U_r,x_train);

for j =1:m
    MSE_t(j) = norm(Mu_full{1}(:,j) - X_full{1}(:,j))/norm(X_full{1}(:,j));
end

MSE = mean(MSE_t);

figure

s_int = 2/127;
x = 0:s_int:2;   % 128 uniform spatial degree
T = 2; 
t_int = T/m;
t1 = 0:t_int:T;  % time discretization 

% DMD
subplot(1,3,1)
[x,t1] = meshgrid(x,t1);
mesh(t1,x,X_full{1}');
xlabel('t');
ylabel('s');
% zlabel('x(s,t)');
title('True solution');
view(75,50); 

subplot(1,3,2)
mesh(t1,x,Mu_full{1}');
xlabel('t');
ylabel('s');
% zlabel('x(s,t)');
title('Predicted mean');
view(75,50); 

subplot(1,3,3)
mesh(t1,x,abs(X_full{1}'-Mu_full{1}'));
xlabel('t');
ylabel('s');
% zlabel('x(s,t)');
title('Predicted error');
view(75,50); 

