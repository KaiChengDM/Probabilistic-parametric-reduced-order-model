function [X1 X2 X_test] = Burger_snapshots(parameter,m)

% Generate snapshot matrix of 1-d heat equation  with periodic boundary
% fluctuation 
%%   Grid coordinates  

Re = parameter;  t0 = exp(Re/8);

g = @(x) (x(:,1)./(x(:,2)+1))./(1 + sqrt((x(:,2)+1)./t0)*exp(Re*x(:,1).^2./(4*x(:,2)+4)));   % Exact solution of Burger equation 

s_int = 2/127;

s = 0:s_int:2;   % 128 uniform spatial degree

T = 2; 

t_int = T/m;

t = 0:t_int:T;   % 128 uniform spatial degree

for i = 1:length(s)
    for j = 1:length(t)
         Snapshot(i,j) = g([s(i) t(j)]); 
    end
end
 

X1 = Snapshot(:,1:m);  X2 = Snapshot(:,2:m+1);  X_test = Snapshot(:,m+2:end); 

end