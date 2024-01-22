
clc; clear ; 

%% collect snapshots

Re = 800;  t0 = exp(Re/8);
g  = @(x) (x(:,1)./(x(:,2)+1))./(1 + sqrt((x(:,2)+1)./t0)*exp(Re*x(:,1).^2./(4*x(:,2)+4)));   % Exact solution of Burger equation 

s_int = 2/127;
s     = 0:s_int:2;      % 128 uniform spatial degree
T     = 2;              % time 
t_int = T/100;
t     = 0:t_int:T;      % 101 time instants

for i = 1:length(s)
    for j = 1:length(t)
         Snapshot(i,j) = g([s(i) t(j)]);  % snapshots
    end
end

m = 89; 

X1 = Snapshot(:,1:m);  X2 = Snapshot(:,2:m+1);  % training set

X_test = Snapshot(:,m+2:end);                   % test set

%% DMD prediction of future state 

threshold = 0.99999;

[Phi,W_r,lambda,b,Xdmd,Atilde,U_r,S_r,V_r,Xdmd_r,Sigma] = DMD_discrete(X1,X2,threshold);  

mm = size(X_test,2);

for i = 1:m
   recon_error(i) = norm(Xdmd(:,i+1) - X2(:,i))./norm(X2(:,i));  % reconstruction error
end

for k = 1:mm
   time_pred(:,k) = lambda.^(k+m).*b;        % predict future states
end

Xdmd_pred = real(Phi * time_pred);

for i = 1:mm
     error(i) = norm(Xdmd_pred(:,i) - X_test(:,i))./norm(X_test(:,i)); % prediction error
end

Error = norm(Xdmd_pred- X_test ,'fro')/norm(X_test ,'fro')

%%  GPR- mixed kernel 

% hyper-parameter of GPR
hyperpar.corr_fun      = 'corrgaussian';
% hyperpar.corr_fun    = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts   = 5;

% training GPR model
X_train = [X1 X2(:,end)];
ROM_Kriging = ROM_Kriging_train_mixed(X_train,threshold,hyperpar);  

% Recover training data
Xtest = X1;
[recon_Mu,recon_Var] = ROM_Kriging_predictor_mixed(Xtest,ROM_Kriging,m);  

for i = 1:m
    recon_error1(i) = norm(recon_Mu(:,i) - X2(:,i))./norm(X2(:,i)); % reconstruction error
end

% predict future state 
Xtest = X2(:,end);
for i = 1:mm           % Auto-regression
    [Mu(:,i),Var(:,i)] = ROM_Kriging_predictor_mixed(Xtest,ROM_Kriging,1); 
    Xtest = Mu(:,i);
end

for i = 1:mm
    error1(i) = norm(Mu(:,i) - X_test(:,i))./norm(X_test(:,i)); % relative error
end

Error1 = norm(Mu - X_test,'fro')/norm(X_test,'fro')

%% GPR  - Gaussian kernel
% hyper-parameter of GPR
hyperpar.corr_fun      = 'corrgaussian';
% hyperpar.corr_fun    = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 5;

% training GPR model
ROM_Kriging1  = ROM_Kriging_train_single(X_train,threshold,hyperpar); 

% Recover training data
Xtest = X1;
[recon_Mu1,recon_Var1] = ROM_Kriging_predictor_single(Xtest,ROM_Kriging1,m);  % reconstruction error

for i = 1:m
    recon_error2(i) = norm(recon_Mu1(:,i) - X2(:,i))./norm(X2(:,i)); 
end

% predict future state 
Xtest = X2(:,end);
for i = 1:mm    % Auto-regression
    [Mu1(:,i),Var1(:,i)] = ROM_Kriging_predictor_single(Xtest,ROM_Kriging1,1); 
    Xtest = Mu1(:,i);
end

for i = 1:mm
    error2(i) = norm(Mu1(:,i) - X_test(:,i))./norm(X_test(:,i)); % relative error
end

Error2 = norm(Mu1 - X_test,'fro')/norm(X_test,'fro')

%% POD - GPR 
% hyper-parameter of GPR
%hyperpar.corr_fun     = 'corrgaussian';
hyperpar.corr_fun      = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts   = 5;

% training GPR model
X_train = [X1 X2(:,end)];
ROM_Kriging2  = POD_Kriging_train(X_train,threshold,hyperpar);

% Recover training data
for i = 1:m+1
    [recon_Mu2(:,i),recon_Var2(:,i)] =  POD_Kriging_predictor(i,ROM_Kriging2); 
    recon_error3(i) = norm(recon_Mu2(:,i) - X_train(:,i))./norm(X_train(:,i));  % reconstruction error
end

recon_error3(1) = [];

% predict future state 
for i = 1:mm
    [Mu2(:,i),Var2(:,i)] = POD_Kriging_predictor(m+i+1,ROM_Kriging2); 
    error3(i) = norm(Mu2(:,i) - X_test(:,i))./norm(X_test(:,i));
end

Error4 = norm(Mu2 - X_test,'fro')/norm(X_test,'fro')

%% comparison of different methods

DMD_error  = [recon_error error] ;
GPR_error1 = [recon_error1 error1] ;
GPR_error2 = [recon_error2 error2] ;
POD_error  = [recon_error3 error3] ;

plot((1:mm+m)*0.02,DMD_error,'--','LineWidth',1.5); hold on 
plot((1:mm+m)*0.02,GPR_error1,':','LineWidth',1.5); hold on
plot((1:mm+m)*0.02,GPR_error2,'-','LineWidth',1.5); hold on
plot((1:mm+m)*0.02,POD_error,'-.','LineWidth',1.5); hold on

legend('DMD','GPR-Mixed kernel','GPR-Gaussian kernel','POD-GPR')
xlabel('t');
ylabel('RE');

%% figure

x = 0:s_int:2;    % 128 uniform spatial degree
t1 = 0:t_int:T;   % 101 time instants
X_full = [X1 X2(:,end) X_test];
figure
subplot(2,2,1)
[x,t1] = meshgrid(x,t1);
mesh(x,t1,X_full');
xlabel('s');
ylabel('t');
view(75,50); 
title('True solution');

subplot(2,2,2);
X_full = [Xdmd  Xdmd_predictor];
mesh(x,t1,real(X_full'));
xlabel('s');
ylabel('t');
% zlabel('x(s,t)');
title('Linear Kernel (DMD)');
view(75,50); 

subplot(2,2,3);
X_full = [X1(:,1) recover_Mean1 Mean1];
mesh(x,t1,X_full');
xlabel('s');
ylabel('t');
% zlabel('x(s,t)');
title('Mixed kernel');
view(75,50); 

subplot(2,2,4);
X_full = [X1(:,1) recover_Mean2 Mean2];
mesh(x,t1,X_full');
xlabel('s');
ylabel('t');
% zlabel('x(s,t)');
title('Guassian Kernel');
view(75,50); 
