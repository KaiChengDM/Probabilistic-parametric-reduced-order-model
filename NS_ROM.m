
clc; clear all;

%% Collecting Snapshots of N-S equation

par  = 500; m = 239;

[X1 X2 X_test] = NS_snapshots(par,m);

X_train = [X1 X2(:,end)];

mt = size(X_test,2);

%%  DMD

threshold = 0.999; 
tic
  [Phi,W_r,lambda,b,Xdmd,Atilde,U_r,S_r,V_r,Xdmd_r,Sigma] = DMD_discrete(X1,X2,threshold);  %% Select the rank that minimizes the recover error of snapshots matrix
toc

RMSE_t = sum(abs(X_train-Xdmd).^2).^(1/2)./sum(abs(X_train).^2).^(1/2);      % reconstruction error 

% DMD prediction of future state 

mm = size(X_test,2);

for i = 1:m
   recon_error(i) = norm(Xdmd(:,i+1) - X2(:,i))./norm(X2(:,i));
end

for k = 1:mm
   time_predictor1(:,k) = lambda.^(k+m).*b;
end

Xdmd_predictor = Phi * time_predictor1;

for i = 1:mm
     error(i) = norm(Xdmd_predictor(:,i) - X_test(:,i))./norm(X_test(:,i));
end

Error = norm(Xdmd_predictor- X_test ,'fro')/norm(X_test ,'fro')

%%  Kriging - mixed kernel 

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

%% Kriging - single kernel

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

%hyperpar.corr_fun     = 'corrgaussian';
hyperpar.corr_fun      = 'corrbiquadspline';
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

plot((1:mm+m)*0.015,GPR_error1,':','LineWidth',1.5); hold on
plot((1:mm+m)*0.015,GPR_error2,'-','LineWidth',1.5); hold on
plot((1:mm+m)*0.015,POD_error,'-.','LineWidth',1.5); hold on
plot((1:mm+m)*0.015,DMD_error,'--','LineWidth',1.5); hold on 

legend('GPR-Mixed kernel','GPR-Gaussian kernel','POD-GPR','DMD')
xlabel('t');
ylabel('RE');

subplot(1,2,2)
GPR_cov1  = [recon_cov1 cov1] ;
GPR_cov2  = [recon_cov2 cov2] ;
POD_cov3  = [recon_cov3 cov3] ;

plot((1:mm+m)*0.015,GPR_cov1,':','LineWidth',1.5); hold on 
plot((1:mm+m)*0.015,GPR_cov2,'-','LineWidth',1.5); hold on
plot((1:mm+m)*0.015,POD_cov3,'-.','LineWidth',1.5); hold on

legend('GPR-Mixed kernel','GPR-Gaussian kernel','POD-GPR')
xlabel('t');
ylabel('Cov');


%% Figure 
% 
%  SCENARIO ='scenario_sphere2.png'; %<--- file (image) that is scenario for navier stokes fluid simulation
%       % SCENARIO ='scenario_driven_lid.png'; %<--- file (image) that is scenario for navier stokes fluid simulation
% 
%         domainX  = 2;                 % length of domain (x-axis) [unit: meters] 
%         xinc     = 200;               % number of nodes across x-component of domain (number of nodes from x=0 to x=domainX); where dx=domainX/xinc (=dy=dn)
%         dt       = 0.001;             % set set delta time [unit: seconds]
%         MI       = 5000;               % number of time steps to perform calculations [time(at time step)=MI*dt]
%         velyi    =  0;                % y-component velocity of region with constant velocity (regions coloured red in scenario image)  [unit: meters/second]
%                                       % [velyi>0,velocity has vector -y with mag abs(velyi) and velyi<0, vel has vector +y with mag of abs(velyi)]
%         velxi    =  1;                % x-component velocity of region with constant velocity (regions coloured red in SCENARIO)   [unit: meters/second]
%                                       % [velxi>0,velocity has vector +x with mag abs(velxi) and velxi<0, vel has vector -x with mag of abs(velxi)]
%         dens     =  1;                % density  [unit: kg/m^3] , water(pure)=1000 blood=1025 air~1 
% %       mu       =  1/600;            % dynamic viscosity [kg/(m*s)]
%         mu       =  par(1);
%     %Poisson Pressure solver parameters!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%         error    = 0.001;             % set tolerance of error for convergence poisson solver of Pressure field (good value to start with is 0.001; which is for most incompressible applications)
%         MAXIT    = 1000;              % maximum number of iterations allowed for poisson solver (increasing this will allow for further convergence of p solver)
%         MINIT    = 1;                 % mininum number of iterations allowed for poisson solver (increasing this will allow for further convergence of p solver)
% 
%     % Note that: MINIT should be less than MAXIT
%     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% 
%     %save parameters $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%     spacelim = 5;     % limit for hardrive space usage (in gigabytes) for externally saved data (x and y component velocity at each time step)
%     ST       = [100 100 500]; % FOR variables of dimensions of ST(1)xST(2) 
%                       % save variable data for x and y component velocities
% 
% % EVERYTHING BELOW HERE IS AUTOMATIC CALCULATIONS/OPERATIONS (NO EDITING REQUIRED)$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
% 
% %Extract scenario from image@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% 
%     [velinit bounds outflow BLUE XI YI] = imagegrab(SCENARIO,xinc); % 
% 
%     dn      =  domainX/XI;            % dn=dx=dy (distance between nodes) [same result as domainY/YI]
%     domainY = domainX.*(YI./XI);      % domain length of y axis (domainY); 
% 
%     strmg1=sprintf('Rendered in Matlab with code created by Jamie M. Johns\n density=%.2fkg/m^3; \\mu=%.4fkg/(m*s); dt=%.4fs, resolution:%.0fx%.0f [for calculations]',dens,mu,dt,XI,YI);
% 
%     [x,y]=meshgrid(linspace(0,domainX,XI),linspace(0,domainY,YI)); % x and y coordinates for u and v
% 
% 
% % PLOT VELOCITY OF LAST CALCULATED FRAME$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
% 
%        %from section 1: LRFX= last calculated u and LRFX= last calculated v
%     LRFX((bounds)==1)=nan; % set nodes inside solid wall region to nan (does not show up in velocity plot)
%     LRFY((bounds)==1)=nan; % set nodes inside solid wall region to nan (does not show up in velocity plot)
% %     F=flipud(sqrt(LRFX.^2+LRFY.^2)); % velocity magnitude
% 
% %% figure 1
% 
%     subplot(2,1,1)
%     F = X_test(:,end); F = reshape(F,YI,XI);
%     T0     = 1; 
%     hold on %hold all objects created on figure (create new object without deleting previous)
%     pcolor(x,y,F); %plot velocity magnitude for each node w.r.t x and y position
%     xlabel('x(meters)') %label x axis
%     ylabel('y(meters)') % label y axis
%     title('True solution') %title for figure (show time to 4 decimal places [%.4f])
%     cb=colorbar; %create colorbar object with reference "cb"
%     cba=caxis; %record current min/max values of colour bar (and hence min/max magnitude of velocity)
%     ylabel(cb,'Velocity (m/s)') %add ylabel to colour bar
%     shading interp %use smooth shading for pcolor (comment out this line to see the difference).
% 
% %     hd=streamslice(x,y,flipud(LRFX),-flipud(LRFY),arrowdensity); %plot streamslice (indicate velocity field)
% %     set(hd,'color',[0.0 0.0 0.0],'linewidth',arrowwidth) %set colour and line width of arrows in streamslice
% %    
%     sld=surf(x,y,ones(size(x)).*-1); %create foreground object (used to highlight solid boundaries with black colour)
%     shading interp %reinforce smooth shading for pcolor and surf objects
%     axis image %stretch axis to scale image
%     set(sld,'facecolor',[0.0 0.0 0.0]) %set colour of sld to black
%     caxis([cba(1) cba(2)]) %reset
% 
%     subplot(2,1,2)
%     F = real(Xdmd_predictor(:,end)); F = reshape(F,YI,XI);
%     T0     = 1; 
%     hold on %hold all objects created on figure (create new object without deleting previous)
%     pcolor(x,y,F); %plot velocity magnitude for each node w.r.t x and y position
%     xlabel('x(meters)') %label x axis
%     ylabel('y(meters)') % label y axis
%     title('GPR-linear kernel') %title for figure (show time to 4 decimal places [%.4f])
%     cb=colorbar; %create colorbar object with reference "cb"
%     cba=caxis; %record current min/max values of colour bar (and hence min/max magnitude of velocity)
%     ylabel(cb,'Velocity (m/s)') %add ylabel to colour bar
%     shading interp %use smooth shading for pcolor (comment out this line to see the difference).
% 
% %     hd=streamslice(x,y,flipud(LRFX),-flipud(LRFY),arrowdensity); %plot streamslice (indicate velocity field)
% %     set(hd,'color',[0.0 0.0 0.0],'linewidth',arrowwidth) %set colour and line width of arrows in streamslice
% %    
%     sld=surf(x,y,ones(size(x)).*-1); %create foreground object (used to highlight solid boundaries with black colour)
%     shading interp %reinforce smooth shading for pcolor and surf objects
%     axis image %stretch axis to scale image
%     set(sld,'facecolor',[0.0 0.0 0.0]) %set colour of sld to black
%     caxis([cba(1) cba(2)]) %reset
