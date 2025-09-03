
%Q1 Parameters
alpha_1 = 0.4;
delta_1 = 0.03;
beta_1 = 0.95;

n = 7; % Number of Chebyshev nodes
sigma = 2; % Risk aversion coefficient

u_1 = @(c) (c^(1 - sigma)) / (1 - sigma);

f_1= @(k) k.^alpha+(1-delta)*k;

param = [alpha_1,beta_1];
a0 = [1.3207;-0.5356; -0.2; 0.1421; -0.0408;0.0114;-0.001];

global kmax;
global kmin;

kmax = 0.1*kss;
kmin = 1.2*kss;

l = 1:1:n;
z = -cos((2*l- 1)*pi/(2*n))';  % Chebyshev nodes for 7 points
k = ((z+ 1).* (kmax-kmin) / 2 + kmin); % Chebyshev nodes in [kmin,kmax]
aa = T(k);

bb = g(k,a0);

cc = R(k, a0, param);

options = optimset('Display', 'off', 'TolFun', 1e-5, 'TolX', 1e-5, 'MaxFunEvals', 1e10, 'MaxIter', 5000);
as = fsolve(@(a)real(R(k, a, param)),a0 ,options);

disp('Optimal coefficients (a):');
disp(as);

k_dense = linspace(kmin, kmax, 100)';
g_dense = g(k_dense, as);

% Plot g(k, a)
figure;
plot(k_dense, g_dense, 'b-', 'LineWidth', 2);
xlabel('k (Capital)');
ylabel('g(k, a) (Policy Function)');
title('Policy Function Approximation');
grid on;

% Function to compute Chebyshev polynomial matrix
function [T] = T(k)
    global kmax;
    global kmin;    
    xi = 2 * (k - kmin) / (kmax - kmin) - 1; % Rescale to [-1, 1]
    n = 7; % Number of Chebyshev polynomials
    T = zeros(length(xi),n); % Preallocate matrix for Chebyshev polynomials
    T(:, 1) = 1; % T_0(x) = 1
    T(:, 2) = xi; % T_1(x) = x
    for i = 3:n
        T(:, i) = 2 * xi .* T(:, i - 1) - T(:, i - 2); % T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    end
end

function [g_k] = g(k, a)
    T_k = T(k); % Compute Chebyshev matrix
    disp(size(T_k))
    disp(size(a))
    g_k = T_k * a; % Policy function approximation
end

% Residual function R(k, a)
function [R] = R(k, a, param)
    % Unpack parameters
    aalpha = param(1);
    bbeta = param(2);
    % Policy function g(k, a)
    g_k = g(k, a); % Evaluate g(k, a)
    g_k_prime = g(g_k, a); % Evaluate g(g(k, a), a)
    % Residuals of Euler equation
    R = g_k - g_k.^(1 - aalpha) .* g_k_prime - aalpha * bbeta * (k.^aalpha - g_k);
end