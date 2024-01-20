clear; clc; close all;

% High-dimensional test example

%% Initial sampling 0.0051 30 samples

d = 10;   N = 20;

mu = zeros(d,1); sig = ones(d,1); 

par = lhsnorm(mu,diag(sig.^2),N);  

m = 169; threshold = 0.99999;

%% Adaptive sampling 1

N1 = 5000; 

p_pool = lhsnorm(mu,diag(sig.^2),N1);  

x_train = par;  

[Mu_t,Var_t,U_r, X_test, ROM_Kriging] = GPR_ROM(x_train,@Redi_10d_snapshots,m,threshold); % train GPR model 
 
[MSE_test,MSE_test1,Var_test,Var_test1] = GPR_ROM_Interpolation(p_pool,@Redi_10d_snapshots_test,Mu_t,Var_t,U_r,x_train);

MSE_m(1) = mean(MSE_test); MSE_m1(1) = mean(MSE_test1); Var_m(1) = mean(Var_test);Var_m1(1) = mean(Var_test1);

%% Adaptive sampling

for i = 1:30
 
    [Var Delta] = Adaptive_GPR_ROM(x_train,p_pool,Mu_t,Var_t);

    Var1 = Var./max(Var);  Delta1 = Delta./max(Delta);

    [value ind] = max(Var1 + Delta1);  % select new parameter 
    
    x_train = [x_train; p_pool(ind,:)];

    [Mu_t,Var_t,U_r,X_test, ROM_Kriging] = GPR_ROM(x_train,@Redi_10d_snapshots,m,threshold); 
    
    [MSE_test,MSE_test1,Var_test,Var_test1] = GPR_ROM_Interpolation(p_pool,@Redi_10d_snapshots_test,Mu_t,Var_t,U_r,x_train);

    MSE_m(i+1)  = mean(MSE_test)
    MSE_m1(i+1) = mean(MSE_test1);
    Var_m(i+1)  = mean(Var_test)
    Var_m1(i+1) = mean(Var_test1);

end


% figure
% plot(x_train(1:N,1),x_train(1:N,2),'mo'); hold on
% plot(x_train(N+1:end,1),x_train(N+1:end,2),'ro')
% % plot(p_pool(ind,1),p_pool(ind,2),'ro')

subplot(1,2,1)
% plot(0.9+(1:N1)*0.1,MSE,'-o','LineWidth',1.5); hold on 
plot(0:3:30,MSE_m(1:3:31),'-o','LineWidth',1.5); hold on 
ylabel('AMRE');
xlabel('Iterations');
% legend('','Initial samples','Enriched samples')

% subplot(1,3,2)
% % plot(0.9+(1:N1)*0.1,MSE,'-o','LineWidth',1.5); hold on 
% plot(1:21, MSE_m1,'-o','LineWidth',1.5); hold on 
% ylabel('AMRE2');
% xlabel('Iterations');
% legend('','Initial samples','Enriched samples')

subplot(1,2,2)
plot(0:3:30,Var_m(1:3:31),'-o','LineWidth',1.5); hold on 
ylabel('AMRStd');
xlabel('Iterations');

%% single paramter prediction

x_test = lhsnorm(mu,diag(sig.^2),1);

[Mean_full,Variance_full,X_full] = GPR_ROM_prediction(x_test,@Redi_10d_snapshots_test,Mu_t,Var_t,U_r,x_train);

for j =1 :200
     MSE_t(j) = norm(Mean_full{1}(:,j) - X_full{1}(:,j))/norm(X_full{1}(:,j));
end

MSE = mean(MSE_t);

figure
ind = 200:200:40001;  
N=100; 
M=40000;
dx=1/N; dt=1/M;     
x=0:dx:1;   t=0:dt:1;  t1 = t(ind);

% DMD
subplot(1,3,1)
[x,t1] = meshgrid(x,t1);
mesh(t1,x,X_full{1}');
xlabel('t');
ylabel('s');
% zlabel('x(s,t)');
title('True solution');
view(75,50); 


% Kriging-mixed kernel
subplot(1,3,2)
mesh(t1,x,Mean_full{1}');
xlabel('t');
ylabel('s');
% zlabel('x(s,t)');
title('Predicted mean');
view(75,50); 


% Kriging-mixed kernel
subplot(1,3,3)
mesh(t1,x,abs(X_full{1}'-Mean_full{1}'));
xlabel('t');
ylabel('s');
% zlabel('x(s,t)');
title('Predicted error');
view(75,50);


