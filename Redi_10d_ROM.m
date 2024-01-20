
clc; clear ; 

%% Nonliear reaction-diffusion equation 

d = 10;  mu = zeros(d,1); sig = ones(d,1); 

par = lhsnorm(mu,diag(sig.^2),1);   m = 169; 

[X1 X2 X_test] = Redi_10d_snapshots(par,m);

%% DMD prediction of future state 

threshold = 0.99999;

[Phi,W_r,lambda,b,Xdmd,Atilde,U_r,S_r,V_r,Xdmd_r,Sigma] = DMD_discrete(X1,X2,threshold);  %% Select the rank that minimizes the recover error of snapshots matrix

mm = size(X_test,2);

for i = 1:m
   recon_error(i) = norm(Xdmd(:,i+1) - X2(:,i))./norm(X2(:,i));
end

for k = 1:mm
   time_pred(:,k) = lambda.^(k+m).*b;
end

Xdmd_pred = real(Phi * time_pred);

for i = 1:mm
     error(i) = norm(Xdmd_pred(:,i) - X_test(:,i))./norm(X_test(:,i));
end

Error = norm(Xdmd_pred- X_test ,'fro')/norm(X_test ,'fro')

%%  Kriging --- mixed kernel 

hyperpar.corr_fun      = 'corrgaussian';
% hyperpar.corr_fun    = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts   = 5;

X_train = [X1 X2(:,end)];

ROM_Kriging = ROM_Kriging_train_mixed(X_train,threshold,hyperpar); 

% Recover training data

Xtest = X1;
[recon_Mu,recon_Var] = ROM_Kriging_predictor_mixed(Xtest,ROM_Kriging,m); 

for i = 1:m
     recon_error1(i) = norm(recon_Mu(:,i) - X2(:,i))./norm(X2(:,i)); 
     recon_cov1(i)   = norm(sqrt(recon_Var(:,i)))/norm(recon_Mu(:,i));
end

Xtest = X2(:,end);

for i = 1:mm           % Auto-regression
    [Mu(:,i),Var(:,i)] = ROM_Kriging_predictor_mixed(Xtest,ROM_Kriging,1); 
    Xtest     = Mu(:,i);
    error1(i) = norm(Mu(:,i) - X_test(:,i))./norm(X_test(:,i));
    cov1(i)   = norm(sqrt(Var(:,i)))/norm(Mu(:,i));
end

Error1 = norm(Mu - X_test,'fro')/norm(X_test,'fro')

%% Kriging  --- Stationary kernel

hyperpar.corr_fun      = 'corrgaussian';
% hyperpar.corr_fun    = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts = 5;

ROM_Kriging1  = ROM_Kriging_train_single(X_train,threshold,hyperpar); 

Xtest = X1;

[recon_Mu1,recon_Var1] = ROM_Kriging_predictor_single(Xtest,ROM_Kriging1,m); 

for i = 1:m
    recon_error2(i) = norm(recon_Mu1(:,i) - X2(:,i))./norm(X2(:,i)); 
    recon_cov2(i)   = norm(sqrt(recon_Var1(:,i)))/norm(recon_Mu1(:,i));
end

Xtest = X2(:,end);

for i = 1:mm    % Auto-regression
    [Mu1(:,i),Var1(:,i)] = ROM_Kriging_predictor_single(Xtest,ROM_Kriging1,1); 
    Xtest     = Mu1(:,i);
    error2(i) = norm(Mu1(:,i) - X_test(:,i))./norm(X_test(:,i));
    cov2(i)   = norm(sqrt(Var1(:,i)))/norm(Mu1(:,i));
end


Error2 = norm(Mu1 - X_test,'fro')/norm(X_test,'fro')

%% POD - Kriging

hyperpar.corr_fun     = 'corrgaussian';
%hyperpar.corr_fun      = 'corrbiquadspline';
hyperpar.opt_algorithm = 'Hooke-Jeeves';
hyperpar.multistarts   = 5;

X_train = [X1 X2(:,end)];

ROM_Kriging2  = POD_Kriging_train(X_train,threshold,hyperpar);

for i = 1:m+1
    [recon_Mu2(:,i),recon_Var2(:,i)] =  POD_Kriging_predictor(i,ROM_Kriging2); 
    recon_error3(i) = norm(recon_Mu2(:,i) - X_train(:,i))./norm(X_train(:,i)); 
    recon_cov3(i)   = norm(sqrt(recon_Var2(:,i)))/norm(recon_Mu2(:,i));
end

recon_error3(1) = []; recon_cov3(1)= [];

for i = 1:mm
    [Mu2(:,i),Var2(:,i)] = POD_Kriging_predictor(m+i+1,ROM_Kriging2); 
    error3(i) = norm(Mu2(:,i) - X_test(:,i))./norm(X_test(:,i));
    cov3(i)   = norm(sqrt(Var2(:,i)))/norm(Mu2(:,i));
end

Error4 = norm(Mu2 - X_test,'fro')/norm(X_test,'fro')

%% Figure
figure
subplot(1,2,1)
DMD_error  = [recon_error error] ;
GPR_error1 = [recon_error1 error1] ;
GPR_error2 = [recon_error2 error2] ;
POD_error  = [recon_error3 error3] ;

plot((1:mm+m)*0.005,GPR_error1,':','LineWidth',1.5); hold on
plot((1:mm+m)*0.005,GPR_error2,'-','LineWidth',1.5); hold on
plot((1:mm+m)*0.005,POD_error,'-.','LineWidth',1.5); hold on
plot((1:mm+m)*0.005,DMD_error,'--','LineWidth',1.5); hold on 

legend('GPR-Mixed kernel','GPR-Gaussian kernel','POD-GPR','DMD')
xlabel('t');
ylabel('RE');

subplot(1,2,2)
cov1  = [recon_cov1 cov1] ;
cov2  = [recon_cov2 cov2] ;
cov3  = [recon_cov3 cov3] ;

plot((1:mm+m)*0.005,cov1,':','LineWidth',1.5); hold on 
plot((1:mm+m)*0.005,cov2,'-','LineWidth',1.5); hold on
plot((1:mm+m)*0.005,cov3,'-.','LineWidth',1.5); hold on

legend('GPR-Mixed kernel','GPR-Gaussian kernel','POD-GPR')
xlabel('t');
ylabel('Cov');

% x = 0:s_int:2;   % 128 uniform spatial degree
% t1 = 0:t_int:T;   % 128 uniform spatial degree
% X_full = [X1 X2(:,end) X_test];
% figure
% subplot(2,2,1)
% [x,t1] = meshgrid(x,t1);
% mesh(x,t1,X_full');
% xlabel('s');
% ylabel('t');
% view(75,50); 
% title('True solution');
% 
% subplot(2,2,2);
% X_full = [Xdmd  Xdmd_predictor];
% mesh(x,t1,real(X_full'));
% xlabel('s');
% ylabel('t');
% % zlabel('x(s,t)');
% title('Linear Kernel (DMD)');
% view(75,50); 
% 
% subplot(2,2,3);
% X_full = [X1(:,1) recover_Mean1 Mean1];
% mesh(x,t1,X_full');
% xlabel('s');
% ylabel('t');
% % zlabel('x(s,t)');
% title('Mixed kernel');
% view(75,50); 
% 
% subplot(2,2,4);
% X_full = [X1(:,1) recover_Mean2 Mean2];
% mesh(x,t1,X_full');
% xlabel('s');
% ylabel('t');
% % zlabel('x(s,t)');
% title('Guassian Kernel');
% view(75,50); 
