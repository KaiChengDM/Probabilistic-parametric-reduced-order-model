function [Phi,W_r,lambda,b,Xdmd,Atilde,U_r,S_r,V_r,Xdmd_r,Sigma] = DMD_discrete(X1,X2,threshold)
%% Computes the Dynamic Mode Decomposition of X1, X2
%
% INPUTS: 
% X1 = X, data matrix
% X2 = X', shifted data matrix
% Columns of X1 and X2 are state snapshots 
% r = target rank of SVD
% dt = time step advancing X1 to X2 (X to X')
%
% OUTPUTS:
% Phi, the DMD modes
% omega, the continuous-time DMD eigenvalues
% lambda, the discrete-time DMD eigenvalues
% b, a vector of magnitudes of modes Phi
% Xdmd, the data matrix reconstrcted by Phi, omega, b

%% DMD

mm1 = size(X1, 2); 

[U, S, V] = svd(X1, 'econ');

Sigma = diag(S);

for i = 1 :length(Sigma)
    energy(i) = sum(Sigma(1:i))./sum(Sigma);
    if energy(i)  > threshold
        r = i;
        break;
    end
end

%  r = min(r, size(U,2));
%  r = rank;

U_r = U(:, 1:r);                % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);

Atilde = U_r' * X2 * V_r / S_r;    % low-rank dynamics
[W_r, D] = eig(Atilde);
Phi = X2 * V_r / S_r * W_r /D;     % DMD modes

lambda = diag(D);                  % discrete-time eigenvalues

%% Compute DMD mode amplitudes b

x1 = X1(:, 1);   
b = Phi\x1;

%% Optimal DMD mode amplitudes b
 
% V_and = ones(r,1);
% 
% for i = 1:mm1-1
%    V_and = [V_and  lambda.^i];
% end
% 
% P = (W_r'*W_r).*conj(V_and*V_and');
% 
% Q = conj(diag(V_and*V_r*S_r'*W_r));
% 
% b =  P\Q;

%% DMD reconstruction

time_dynamics = zeros(r, mm1+1);

% t = (0:mm1-1)*dt; % time vector

for iter = 1:mm1+1
    time_dynamics(:,iter) = b.*lambda.^(iter-1);
end

Xdmd = Phi * time_dynamics;

%% Low-dimensional State

% for iter = 1:mm1
%     time_dynamics(:,iter) = b.*lambda.^iter;
% end
% 
% Xdmd_r = W_r*time_dynamics;

Xdmd_r = U_r'*Xdmd;


end

