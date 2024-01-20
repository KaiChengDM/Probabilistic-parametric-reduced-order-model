function ROM_Kriging = POD_Kriging_train(X,threshold,hyperpar)

% Training reduced order model with Kriging 

[U, S, V] = svd(X, 'econ');

Sigma = diag(S);

for i = 1 :length(Sigma)
    energy(i) = sum(Sigma(1:i))./sum(Sigma);
    if energy(i)  > threshold
        r = i;
        break;
    end
end

r = min(r, size(U,2));

U_r = U(:, 1:r);                % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);

X_r = U_r'*X;  % Coefficient 

n = 1;  m = size(X_r,2); 

u_input = (1:m)';    u_output = X_r';

N = length(u_input); 

hyperpar.theta = 1.*ones(1,n);
hyperpar.lb    = 10^-7.*ones(1,n);
hyperpar.ub    = 50*ones(1,n);

mean_output = mean(u_output);   
std_output  = std(u_output);

lb_input    = min(u_input);   
ub_input    = max(u_input); 

u_input  = (u_input  - repmat(lb_input,N,1))./(repmat(ub_input,N,1)-repmat(lb_input,N,1));
u_output = (u_output - repmat(mean_output,N,1))./repmat(std_output,N,1);

for i = 1:r
   
    x = u_input; 
    y = u_output(:,i);  
    
    % plot(x,y)
    
    inputpar.x = x;
    inputpar.y = y;

    ROM_Kriging{i} = Kriging_fit_single(inputpar,hyperpar);  % training Kriging model 
    
    % [Mean Variance] = Kriging_predictor_single(x,ROM_Kriging{i}); % assign same weight for each component of y_r
    % 
    % % norm(y- Mean)./norm(y)
    % plot(x,y,'m:'); hold on
    % plot(x,Mean,'r-'); hold on
    % plot(x,y,'ro')

    ROM_Kriging{i}.basis = U_r;  
    ROM_Kriging{i}.dim = n; 
    ROM_Kriging{i}.rank = r;
    ROM_Kriging{i}.lb_input   = lb_input;
    ROM_Kriging{i}.ub_input   = ub_input;
    ROM_Kriging{i}.mean_output  = mean_output;
    ROM_Kriging{i}.std_output   = std_output;

 end

%  ROM_Kriging.basis = U_r;  
%  ROM_Kriging.Yr = u_output;  % multiple output
% 
% %  ROM_Kriging.mean_input = mean_input;
% %  ROM_Kriging.std_input  = std_input;
%  ROM_Kriging.lb_input   = lb_input;
%  ROM_Kriging.ub_input   = ub_input;
%  ROM_Kriging.dim = n; 

end