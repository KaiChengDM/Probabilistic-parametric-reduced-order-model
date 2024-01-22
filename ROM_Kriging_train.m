function ROM_Kriging = ROM_Kriging_train(X_r,hyperpar)

% Training reduced order model with Kriging --- mixed kernel

[n,m] = size(X_r);

hyperpar.theta = [1 0.1]; 
hyperpar.lb    = [10^-5 10^-7];
hyperpar.ub    = [10 1];

ub_data  = max(abs(X_r'));         % upper bound of each component

X_r1 = X_r'./repmat(ub_data,m,1);  % normalized reduced snapshots

u_input = X_r1(1:m-1,:);    u_output = X_r1(2:m,:);

 for k = 1:n

     inputpar.x = u_input;
     inputpar.y = u_output(:,k);

     t1 = clock;
        ROM_Kriging{k} = Kriging_fit(inputpar,hyperpar);  % training Kriging model 
     t2 = clock;

     ROM_Kriging{k}.Yr = inputpar.y;  % multiple output

 end

  ROM_Kriging{1}.ub_data = ub_data;

end