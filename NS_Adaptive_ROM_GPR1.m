clear; clc;
close all;

%% Initial sampling 

lb = 100;  ub = 500; dim = 1;  N = 3;

par = [lb; ub;(lb+ub)/2];

m = 299; threshold = 0.999;

%% Adaptive sampling 

p_pool = (lb(1):(ub(1)-lb(1))/49:ub(1))'; 
 
x_train = par;  
 
t1 = clock;

[Mu_t,Var_t,U_r, X_test, ROM_Kriging] = GPR_ROM_NS(x_train,@NS_snapshots,m,threshold); % train GPR model 

t2 = clock;

Time = etime(t2,t1);

[MSE_test,MSE_test1,Var_test,Var_test1] = GPR_ROM_Interpolation1(p_pool,@NS_snapshots_test,Mu_t,Var_t,U_r,x_train);

MSE_m(1) = mean(MSE_test); MSE_m1(1) = mean(MSE_test1); Var_m(1) = mean(Var_test); Var_m1(1) = mean(Var_test1);

%% adaptive sampling stage

for i = 1:5
 
    i

    t1 = clock;

    [Var Delta] = Adaptive_GPR_ROM1(x_train,p_pool,Mu_t,Var_t);

    Var1 = Var./max(Var);  Delta1 = Delta./max(Delta);

    [value ind] = max(Var1 + Delta1);     % select new parameter 
 
    x_train = [x_train; p_pool(ind,:)];
    
    [Mu_t,Var_t,U_r,X_test, ROM_Kriging] = GPR_ROM_NS1(x_train,@NS_snapshots,m,threshold); 
      
    t2 = clock;

    Time = Time + etime(t2,t1)

    [MSE_test,MSE_test1, Var_test] = GPR_ROM_Interpolation1(p_pool,@NS_snapshots_test,Mu_t,Var_t,U_r,x_train);

    MSE_m(i+1) = mean(MSE_test); MSE_m1(i+1) = mean(MSE_test1); Var_m(i+1) = mean(Var_test);

end

% convergenve curves
subplot(1,2,1)
plot(0:5, MSE_m,'-o','LineWidth',1.5); hold on 
ylabel('AMRE');
xlabel('Iterations');

subplot(1,2,2)
plot(0:5,Var_m,'-o','LineWidth',1.5); hold on 
ylabel('AMRStd');
xlabel('Iterations');


%% single paramter prediction

x_test = 475; 

tic
[Mu_full,Var_full,X_full] = GPR_ROM_prediction1(x_test,@NS_snapshots_test,Mu_t,Var_t,U_r,x_train);
toc

for j =1 :300
     MSE_t(j) = norm(Mu_full{1}(:,j) - X_full{1}(:,j))/norm(X_full{1}(:,j));
end

MSE = mean(MSE_t);

%% figure

 SCENARIO ='scenario_sphere2.png'; %<--- file (image) that is scenario for navier stokes fluid simulation
      % SCENARIO ='scenario_driven_lid.png'; %<--- file (image) that is scenario for navier stokes fluid simulation

        domainX  = 2;                 % length of domain (x-axis) [unit: meters] 
        xinc     = 200;               % number of nodes across x-component of domain (number of nodes from x=0 to x=domainX); where dx=domainX/xinc (=dy=dn)
        dt       = 0.001;             % set set delta time [unit: seconds]
        MI       = 4500;               % number of time steps to perform calculations [time(at time step)=MI*dt]
        velyi    =  0;                % y-component velocity of region with constant velocity (regions coloured red in scenario image)  [unit: meters/second]
                                      % [velyi>0,velocity has vector -y with mag abs(velyi) and velyi<0, vel has vector +y with mag of abs(velyi)]
        velxi    =  1;                % x-component velocity of region with constant velocity (regions coloured red in SCENARIO)   [unit: meters/second]
                                      % [velxi>0,velocity has vector +x with mag abs(velxi) and velxi<0, vel has vector -x with mag of abs(velxi)]
        dens     =  1;                % density  [unit: kg/m^3] , water(pure)=1000 blood=1025 air~1 
%       mu       =  1/600;            % dynamic viscosity [kg/(m*s)]
        mu       = 1/x_test;
    %Poisson Pressure solver parameters!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        error    = 0.001;             % set tolerance of error for convergence poisson solver of Pressure field (good value to start with is 0.001; which is for most incompressible applications)
        MAXIT    = 1000;              % maximum number of iterations allowed for poisson solver (increasing this will allow for further convergence of p solver)
        MINIT    = 1;                 % mininum number of iterations allowed for poisson solver (increasing this will allow for further convergence of p solver)
        
    % Note that: MINIT should be less than MAXIT
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    %save parameters $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    spacelim = 5;     % limit for hardrive space usage (in gigabytes) for externally saved data (x and y component velocity at each time step)
    ST       = [100 100 500]; % FOR variables of dimensions of ST(1)xST(2) 
                      % save variable data for x and y component velocities

% EVERYTHING BELOW HERE IS AUTOMATIC CALCULATIONS/OPERATIONS (NO EDITING REQUIRED)$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%Extract scenario from image@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   
    [velinit bounds outflow BLUE XI YI] = imagegrab(SCENARIO,xinc); % 
    
    dn      =  domainX/XI;            % dn=dx=dy (distance between nodes) [same result as domainY/YI]
    domainY = domainX.*(YI./XI);      % domain length of y axis (domainY); 
       
    strmg1=sprintf('Rendered in Matlab with code created by Jamie M. Johns\n density=%.2fkg/m^3; \\mu=%.4fkg/(m*s); dt=%.4fs, resolution:%.0fx%.0f [for calculations]',dens,mu,dt,XI,YI);

    [x,y]=meshgrid(linspace(0,domainX,XI),linspace(0,domainY,YI)); % x and y coordinates for u and v


% PLOT VELOCITY OF LAST CALCULATED FRAME$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

     figure(1) %plot of velocity for last calculated time step (of section 1)
    %from section 1: LRFX= last calculated u and LRFX= last calculated v
    LRFX((bounds)==1)=nan; % set nodes inside solid wall region to nan (does not show up in velocity plot)
    LRFY((bounds)==1)=nan; % set nodes inside solid wall region to nan (does not show up in velocity plot)

%     F=flipud(sqrt(LRFX.^2+LRFY.^2)); % velocity magnitude

%    F = Mean_full{1}(:,end); F = reshape(F,YI,XI);
     F = X_full{1}(:,end); F = reshape(F,YI,XI);

    T0     = 1; 
    hold on %hold all objects created on figure (create new object without deleting previous)
    pcolor(x,y,F); %plot velocity magnitude for each node w.r.t x and y position
    xlabel(sprintf('x(meters)\n[time-step:%.0f of %.0f (last calculated timestep)]',T0,MI)) %label x axis
    ylabel('y(meters)') % label y axis
    title(sprintf('%s\nsimulation time=%.4fseconds',strmg1,T0.*dt)) %title for figure (show time to 4 decimal places [%.4f])
    cb=colorbar; %create colorbar object with reference "cb"
    cba=caxis; %record current min/max values of colour bar (and hence min/max magnitude of velocity)
    ylabel(cb,'Velocity (m/s)') %add ylabel to colour bar
    shading interp %use smooth shading for pcolor (comment out this line to see the difference).
    
%     hd=streamslice(x,y,flipud(LRFX),-flipud(LRFY),arrowdensity); %plot streamslice (indicate velocity field)
%     set(hd,'color',[0.0 0.0 0.0],'linewidth',arrowwidth) %set colour and line width of arrows in streamslice
%    
    sld=surf(x,y,ones(size(x)).*-1); %create foreground object (used to highlight solid boundaries with black colour)
    shading interp %reinforce smooth shading for pcolor and surf objects
    axis image %stretch axis to scale image
    set(sld,'facecolor',[0.0 0.0 0.0]) %set colour of sld to black
    caxis([cba(1) cba(2)]) %reset

    %%

    subplot(3,1,1)
    F = X_full{1}(:,end); F = reshape(F,YI,XI);
    T0     = 1; 
    hold on %hold all objects created on figure (create new object without deleting previous)
    pcolor(x,y,F); %plot velocity magnitude for each node w.r.t x and y position
    xlabel('x(meters)') %label x axis
    ylabel('y(meters)') % label y axis
    title('True solution') %title for figure (show time to 4 decimal places [%.4f])
    cb=colorbar; %create colorbar object with reference "cb"
    cba=caxis; %record current min/max values of colour bar (and hence min/max magnitude of velocity)
    ylabel(cb,'Velocity (m/s)') %add ylabel to colour bar
    shading interp %use smooth shading for pcolor (comment out this line to see the difference).
    
%     hd=streamslice(x,y,flipud(LRFX),-flipud(LRFY),arrowdensity); %plot streamslice (indicate velocity field)
%     set(hd,'color',[0.0 0.0 0.0],'linewidth',arrowwidth) %set colour and line width of arrows in streamslice
%    
    sld=surf(x,y,ones(size(x)).*-1); %create foreground object (used to highlight solid boundaries with black colour)
    shading interp %reinforce smooth shading for pcolor and surf objects
    axis image %stretch axis to scale image
    set(sld,'facecolor',[0.0 0.0 0.0]) %set colour of sld to black
    caxis([cba(1) cba(2)]) %reset
    
    subplot(3,1,2)
%   F = real(Xdmd_predictor(:,end)); F = reshape(F,YI,XI);
    F = Mu_full{1}(:,end); F = reshape(F,YI,XI);
    T0     = 1; 
    hold on %hold all objects created on figure (create new object without deleting previous)
    pcolor(x,y,F); %plot velocity magnitude for each node w.r.t x and y position
    xlabel('x(meters)') % l
    ylabel('y(meters)') % label y axis
    title('Predicted solution') %title for figure (show time to 4 decimal places [%.4f])
    cb=colorbar; %create colorbar object with reference "cb"
    cba=caxis; %record current min/max values of colour bar (and hence min/max magnitude of velocity)
    ylabel(cb,'Velocity (m/s)') %add ylabel to colour bar
    shading interp %use smooth shading for pcolor (comment out this line to see the difference).
    
%     hd=streamslice(x,y,flipud(LRFX),-flipud(LRFY),arrowdensity); %plot streamslice (indicate velocity field)
%     set(hd,'color',[0.0 0.0 0.0],'linewidth',arrowwidth) %set colour and line width of arrows in streamslice
%    
    sld=surf(x,y,ones(size(x)).*-1); %create foreground object (used to highlight solid boundaries with black colour)
    shading interp %reinforce smooth shading for pcolor and surf objects
    axis image %stretch axis to scale image
    set(sld,'facecolor',[0.0 0.0 0.0]) %set colour of sld to black
    caxis([cba(1) cba(2)]) %reset
    
    subplot(3,1,3)
%   F = real(Xdmd_predictor(:,end)); F = reshape(F,YI,XI);
    F = X_full{1}(:,end); F = reshape(F,YI,XI);
    F1 = Mu_full{1}(:,end); F1 = reshape(F1,YI,XI);

    T0     = 1; 
    hold on %hold all objects created on figure (create new object without deleting previous)
    pcolor(x,y,abs(F-F1)); %plot velocity magnitude for each node w.r.t x and y position
    xlabel('x(meters)') % l
    ylabel('y(meters)') % label y axis
    title('Predicted error') %title for figure (show time to 4 decimal places [%.4f])
    cb=colorbar; %create colorbar object with reference "cb"
    cba=caxis; %record current min/max values of colour bar (and hence min/max magnitude of velocity)
    ylabel(cb,'Velocity (m/s)') %add ylabel to colour bar
    shading interp %use smooth shading for pcolor (comment out this line to see the difference).
    
%     hd=streamslice(x,y,flipud(LRFX),-flipud(LRFY),arrowdensity); %plot streamslice (indicate velocity field)
%     set(hd,'color',[0.0 0.0 0.0],'linewidth',arrowwidth) %set colour and line width of arrows in streamslice
%    
    sld=surf(x,y,ones(size(x)).*-1); %create foreground object (used to highlight solid boundaries with black colour)
    shading interp %reinforce smooth shading for pcolor and surf objects
    axis image %stretch axis to scale image
    set(sld,'facecolor',[0.0 0.0 0.0]) %set colour of sld to black
    caxis([cba(1) cba(2)]) %reset

