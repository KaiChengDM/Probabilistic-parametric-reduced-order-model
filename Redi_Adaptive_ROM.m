clear; clc; close all;

%% Initial sampling 

lb = [0.5 1]; ub = [1.5 5]; % lower bound and upper bound of two parameter

dim = 2;  N = 5;            % dimension and initial sample size 

pp = sobolset(dim,'Skip',20); u = net(pp,N);  % sampling 

for i = 1:dim
  par(:,i)= u(:,i)*(ub(i)-lb(i))+lb(i); % transform to physical space
end

m = 149; threshold = 0.99999;  % snapshots number and SVD truncation number

%% Initial pROM

N1 = 1000;  pp = sobolset(dim,'Skip',100); u1 = net(pp,N1);  % Test sample pool

for i = 1:dim
   p_pool(:,i)= u1(:,i)*(ub(i)-lb(i))+lb(i);
end

x_train = par;  

% train initial GPR model 
[Mu_t,Var_t,U_r, X_test, ROM_Kriging] = GPR_ROM(x_train,@Redi_snapshots,m,threshold); 
 
% assess the accuracy of initial pROM 
[MSE_test,MSE_test1,Var_test,Var_test1] = GPR_ROM_Interpolation(p_pool,@Redi_snapshots_test,Mu_t,Var_t,U_r,x_train);

MSE_m(1) = mean(MSE_test); MSE_m1(1) = mean(MSE_test1); Var_m(1) = mean(Var_test);Var_m1(1) = mean(Var_test1);


%% Adaptive sampling

for i = 1:15
 
    [Var Delta] = Adaptive_GPR_ROM(x_train,p_pool,Mu_t,Var_t); % compute bais and variance

    Var1 = Var./max(Var);  Delta1 = Delta./max(Delta);

    [value ind] = max(Var1 + Delta1);    % select new parameter 
     
    x_train = [x_train; p_pool(ind,:)];

    % update GPR model
    [Mu_t,Var_t,U_r,X_test, ROM_Kriging] = GPR_ROM(x_train,@Redi_snapshots,m,threshold); 
    
    % assess the accuracy of pROM
    [MSE_test,MSE_test1,Var_test,Var_test1] = GPR_ROM_Interpolation(p_pool,@Redi_snapshots_test,Mu_t,Var_t,U_r,x_train);

    MSE_m(i+1)  = mean(MSE_test)
    MSE_m1(i+1) = mean(MSE_test1)
    Var_m(i+1)  = mean(Var_test);
    Var_m1(i+1) = mean(Var_test1);

end

% convergence curve
subplot(1,2,1)
plot(0:15, MSE_m,'-o','LineWidth',1.5); hold on 
ylabel('AMRE');
xlabel('Iterations');

subplot(1,2,2)
plot(0:15,Var_m,'-o','LineWidth',1.5); hold on 
ylabel('AMRStd');
xlabel('Iterations');

N = 5;
figure
plot(x_train(1:N,1),x_train(1:N,2),'ro','LineWidth',2); hold on
plot(x_train(N+1:end,1),x_train(N+1:end,2),'msquare','LineWidth',2)
xlabel('\omega_1'); 
ylabel('\omega_2');
legend('Initial samples','Enriched samples')


%% Prediction in parameter domain

x1 = lb(1):0.01:ub(1); x2 = lb(2):0.04:ub(2);

nn = length(x1);
for i = 1:length(x1)
   for j = 1:length(x2)
       x_test((i-1)*nn+j,:) = [x1(i) x2(j)];
   end
end
 
[MRE_test2, MRStd_test2] = GPR_ROM_Interpolation(x_test,@Redi_snapshots_test,Mu_t,Var_t,U_r,x_train);

Error = mean(MSE_test2)

for i = 1:length(x1)
   for j = 1:length(x2)
       MRE_ij(i,j)   = MREtest2((i-1)*nn+j);     
       MRStd_ij(i,j) = MRStd_test2((i-1)*nn+j);  
   end
end

[x1,x2] = meshgrid(x1,x2);

subplot(1,2,1)
mesh(x1,x2,MRE_ij); hold on
% plot(x_train(:,1),x_train(:,2),'mo')
xlabel('\omega_1'); 
ylabel('\omega_2');
zlabel('MRE');

subplot(1,2,2)
mesh(x1,x2,MRStd_ij);hold on
% plot(x_train(:,1),x_train(:,2),'mo')
xlabel('\omega_1'); 
ylabel('\omega_2');
zlabel('MRStd');


%% single paramter prediction

x_test = [0.8 3];  % test parameter 

[Mu_full,Var_full,X_full] = GPR_ROM_prediction(x_test,@Redi_snapshots_test,Mu_t,Var_t,U_r,x_train);

for j =1 :200
     MSE_t(j) = norm(Mu_full{1}(:,j) - X_full{1}(:,j))/norm(X_full{1}(:,j));  % relative error
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

