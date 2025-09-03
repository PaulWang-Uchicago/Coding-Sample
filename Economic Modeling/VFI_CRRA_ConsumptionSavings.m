clc; clear all;

%% 1.b
% Parameters
beta = 0.95;
sigma = 2;
r = 0.2;
N = 1000;

% Discretize state space
wgrid = linspace(0.1, 5, N)';
consu = wgrid-wgrid'/(1+r);
consu(consu < 0) = 0; % Restrict > 0

% Utility function
Ret = (1 / (1 - sigma)) * consu.^(1 - sigma);

% Matrix storing value functions
v = zeros(N, N);

% Initial guess for value function
v(:, 1) = (1 / (1 - sigma)) *wgrid .^(1 - sigma);

% Value Function Iteration
for i = 1:N-1
    v(:, i+1) = max((Ret + beta * v(:, i)')');

    % Convergence check
    if max(abs(v(:, i+1) - v(:, i))) <= 1e-20
        disp("Convergence achieved");
        disp(i)
        v(:, i+2:end) = repmat(v(:, i+1), 1, size(v(:, i+2:end), 2)); 
        break;
    end
end

% Ensure valFunc is always assigned (last computed value function)
valFunc = v(:, end);

% Compute policy function index
[~, windex] = max((Ret + beta * valFunc').');

% Compute the policy function
policy_function_w = wgrid(windex); % Maps indices to actual capital values

% Extract optimal consumption using correct indexing
c_value = zeros(N, 1); % Preallocate for efficiency
for i = 1:N
    c_value(i) = consu(i,windex(i));
end

% Analitical function
syms B
theta = (1+r)*beta*B^(-1/sigma)/(1+(1+r)*beta*B^(-1/sigma));
eqn = B == theta^(1-sigma)+beta*B*(1-theta)^(1-sigma)*(1+r)^(-sigma);
sol = solve(eqn, B);
V_a =sol*(1 / (1 - sigma)) *wgrid .^(1 - sigma);
theta_sol = (1+r)*beta*sol^(-1/sigma)/(1+(1+r)*beta*sol^(-1/sigma));
C_a = theta_sol*wgrid;
W_a =(1-theta_sol)*(1+r)*wgrid;

figure;
plot(wgrid, valFunc, 'r', 'Linewidth', 2);
legend('Numerical Approximation', 'Location', 'best');
xlabel('Wealth Grid');  % Label for the x-axis
ylabel('Value Function');  % Label for the y-axis
title('Numerical Approximation of the Value Function'); % Title for the plot
grid on; % Adds a grid for better visibility


%% 1.c
% Compare analytical solution and numerical approximation
figure;
plot(wgrid, valFunc, 'r', wgrid, V_a, 'b', 'Linewidth', 2);
legend('Numerical Approximation', 'Analytical Solution', 'Location', 'best');
xlabel('Wealth Grid');  % Label for the x-axis
ylabel('Value Function');  % Label for the y-axis
title('Comparison of Numerical Approximation and Analytical Solution of Value Function'); % Title for the plot
grid on; % Adds a grid for better visibility


%% 1.d
figure;
plot(wgrid, policy_function_w, 'r',wgrid, W_a,'b', 'LineWidth', 2);
legend('Numerical Approximation', 'Analytical Solution', 'Location', 'best');
xlabel('Wealth Grid');  % Label for the x-axis
ylabel('Policy Function');  % Label for the y-axis
title('Comparison of Numerical Approximation and Analytical Solution of Policy Function'); % Title for the plot
grid on; % Adds a grid for better visibility

% Plot the consumption policy function
figure;
plot(wgrid, c_value ,'r',wgrid, C_a,'b','LineWidth', 2);
legend('Numerical Approximation', 'Analytical Solution', 'Location', 'best');
xlabel('Wealth Grid');  % Label for the x-axis
ylabel('Consumption');  % Label for the y-axis
title('Comparison of Numerical Approximation and Analytical Solution of Consumption'); % Title for the plot
grid on; % Adds a grid for better visibility

%% d)
r     = 0.20;     % interest rate
beta  = 0.95;     % discount factor
sigma = 2;        % CRRA exponent
param = [r, beta, sigma];

n    = 7;         % number of Chebyshev polynomials
kmin = 0.1;       % state space minimum
kmax = 5.0;       % state space maximum

% Chebyshev nodes z in [-1,1], then mapped to w in [kmin,kmax]
l = (1:n)';
z = -cos((2*l - 1)*pi/(2*n));         % n Chebyshev nodes in [-1,1]
w = 0.5*(z + 1)*(kmax - kmin) + kmin; % map each z to [kmin,kmax]

% Analytical fraction (1-alpha):
alpha = 1 - ((beta*(1+r))^(1/sigma)) / (1+r);
M = (1 - alpha)*(1+r);   % next-period wealth multiple

% Compute c and d for the linear function c + d*x
c = M*( kmin + 0.5*(kmax - kmin) );
d = M*( 0.5*(kmax - kmin) );

% Build a0 (length n) with a0(1)=c, a0(2)=d, the rest=0
a0 = zeros(n,1);
a0(1) = c;
a0(2) = d;

% Options for fsolve
options = optimset('Display', 'iter', ...
                   'TolFun',   1e-9, ...
                   'TolX',     1e-9, ...
                   'MaxIter',  1e5, ...
                   'MaxFunEvals',1e6);

%% 2. SOLVE FOR CHEBYSHEV COEFFICIENTS
as = fsolve(@(a) R(w,a,param,kmin,kmax,n), a0, options);

disp(' ');
disp('Optimal Chebyshev coefficients (a):');
disp(as);

%% 3. EVALUATE & PLOT THE POLICY FUNCTION
% Evaluate on a dense grid
wgrid = linspace(kmin, kmax, 200)';
gpol  = g_of_w(wgrid, as, kmin, kmax, n);

% Compare with analytical (closed-form) policy for CRRA + interest r
alpha = 1 - ((beta*(1+r))^(1/sigma)) / (1+r);
policy_analytical = (1 - alpha)*(1+r)*wgrid;

figure('Color','w');
plot(wgrid, policy_analytical, 'r-', 'LineWidth', 2, ...
     'DisplayName','Analytical'); 
hold on;
plot(wgrid, gpol, 'b--', 'LineWidth', 2, ...
     'DisplayName','Collocation Approx.');
xlabel('w (Wealth)');
ylabel('w'' = g(w) (Next Period Wealth)');
title('Orthogonal Collocation vs Analytical Policy');
legend('Location','best');
grid on;

% R(...) = Euler-equation residual at each collocation node
function Rvals = R(w, a, param, kmin, kmax, n)
    r     = param(1);
    beta  = param(2);
    sigma = param(3);

    % Approx policy
    g_w = g_of_w(w, a, kmin, kmax, n);
    % Next period's policy
    g_w2 = g_of_w(g_w, a, kmin, kmax, n);

    % Consumption
    c_t     = w   - g_w /(1+r);
    c_tnext = g_w - g_w2/(1+r);

    % Euler eq. residual
    Rvals = c_t.^(-sigma) - beta*(1+r)* c_tnext.^(-sigma);
end

% g_of_w(w, a) = approximate policy using Chebyshev polynomials
function gvals = g_of_w(w, a, kmin, kmax, n)
    x    = 2*(w - kmin)/(kmax - kmin) - 1;   % map [kmin,kmax] -> [-1,1]
    Tmat = chebT(x, n);                     % Nx n matrix
    gvals = Tmat * a;                       % polynomial approximation
end

% chebT(x, n) = standard Chebyshev polynomials T_0,...,T_{n-1}
function Tmat = chebT(x, n)
    Nx = length(x);
    Tmat = zeros(Nx, n);

    if n >= 1
        Tmat(:,1) = 1;     % T0(x)
    end
    if n >= 2
        Tmat(:,2) = x;     % T1(x)
    end
    for m = 3:n
        Tmat(:,m) = 2.*x.*Tmat(:,m-1) - Tmat(:,m-2);  % Recurrence
    end
end
