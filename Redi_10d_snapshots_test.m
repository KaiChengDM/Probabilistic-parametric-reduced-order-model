function Phi_t = Redi_10d_snapshots_test(par)

% Generate snapshot matrix of reaction-diffusion equation
%%   Grid coordinates  

a   = KL_Coefficient(par(1:5),1,0.2,0.5); 
niu = KL_Coefficient(par(6:end),3,0.5,0.5);

% a = KL_Coefficient(parameter); 
% niu = KL_Coefficient1(parameter(6:end));

N=100; 
M=40000;
dx=1/N; 
dt=1/M;     
x=0:dx:1;  
t=0:dt:1;
r=a.*dt/dx^2; 

%% Initial condition,boundary condition

xl = 0.5 + 0.5*sin(pi*x);

N = length(x)-1;
M = length(t)-1;
Phi = zeros(M+1,N+1);

Phi(1,:) = xl;                
Phi(:,1) = 0.5;
Phi(:,end) = 0.5;

%%  Difference equation

for j=1:M
    for i=2:N
        Phi(j+1,i) = Phi(j,i) + r(i)*(Phi(j,i)-Phi(j,i-1))^2 + Phi(j,i)*r(i)*(Phi(j,i+1)+Phi(j,i-1)-2*Phi(j,i)) -dt*niu(i)*(Phi(j,i)-Phi(j,i)^3);
    end
end

%%   Mapping
% figure  
% [x,t]=meshgrid(x,t);
% mesh(x,t,Phi);
% xlabel('x');
% ylabel('t');
% zlabel('\Phi(x,t)');
% title('\Phi(x,t)');
% view(75,50);  

%%  Snapshot matrix

ind = 200:200:M+1;  

Phi_t = Phi(ind,:)';

end