function [X1 X2 X_test] = Redi_10d_snapshots(par,m)

% Generate snapshot matrix of reaction-diffusion equation

%%   Grid coordinates  

% a = KL_Coefficient(parameter(1:5)); 
% niu = KL_Coefficient1(parameter(6:end));

a   = KL_Coefficient(par(1:5),1,0.2,0.5); 
niu = KL_Coefficient(par(6:end),3,0.5,0.5);

N  = 100; 
M  = 40000;
dx = 1/N; 
dt = 1/M;     
x  = 0:dx:1;  
t  = 0:dt:1;
r  = a.*dt/dx^2; 

%% Initial condition,boundary condition

xl = 0.5 + 0.5*sin(pi*x);

N = length(x)-1;
M = length(t)-1;
Phi = zeros(M+1,N+1);

Phi(1,:)   = xl;                
Phi(:,1)   = 0.5;
Phi(:,end) = 0.5;

%%  Difference equation

for j=1:M
    for i=2:N
        Phi(j+1,i) = Phi(j,i) + r(i)*(Phi(j,i)-Phi(j,i-1))^2 + Phi(j,i)*r(i)*(Phi(j,i+1)+Phi(j,i-1)-2*Phi(j,i)) -dt*niu(i)*(Phi(j,i)-Phi(j,i)^3);
    end
end

%%  Snapshot matrix

ind = 200:200:M+1;  

Phi_t = Phi(ind,:)';

X1 = Phi_t(:,1:m); X2 = Phi_t(:,2:m+1);  X_test = Phi_t(:,m+2:end);

end