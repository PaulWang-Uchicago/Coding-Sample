alpha = 0.36;
delta = 0.03;
beta = 0.98; 
sigma = 2;
lss = 1/3;

kss = ( (alpha * lss^(1-alpha)) / (1/beta - (1 - delta)) )^(1/(1-alpha));
css = kss^alpha *lss^(1-alpha) - delta*kss;
gamma = css / ( css + (1-alpha) * kss^alpha * lss^(-alpha) * (1-lss) );
%% 2.d
% Parameters
alpha = 0.36;
delta = 0.03;
beta = 0.98;
sigma = 2;
l_ss_target = 1/3;
n = 8;

% Part c: Solve for steady-state k, c, and gamma
% ---------------------------------------------------------------------
% Find k_ss using Euler equation
fun_k = @(k) 1/beta - alpha*k^(alpha-1)*l_ss_target^(1-alpha) - (1 - delta);
k_ss = fzero(fun_k, 7);

% Compute F_ℓ and c_ss
F_l = (1 - alpha) * k_ss^alpha * l_ss_target^(-alpha);
c_ss = k_ss^alpha * l_ss_target^(1 - alpha) - delta * k_ss;

% Solve for gamma using intra-temporal condition
fun_gamma = @(gamma) - (1 - gamma)*(2/3)^(gamma - 2) + gamma*(F_l/c_ss)*(3/2)^(1 - gamma);
gamma = fzero(fun_gamma, 0.5);

% Part d: Orthogonal collocation for policy functions
% ---------------------------------------------------------------------
% Define interval [a, b]
a = 0.3 * k_ss;
b = 1.2 * k_ss;

% Generate Chebyshev nodes in [a, b]
j = 1:n;
z_nodes = cos((2*j - 1) * pi / (2*n))';
k_nodes = (z_nodes + 1) * (b - a)/2 + a;

% Parameters for residual function
params.alpha = alpha;
params.delta = delta;
params.beta = beta;
params.sigma = sigma;
params.gamma = gamma;
params.a = a;
params.b = b;
params.k_nodes = k_nodes;
params.n = n;
params.k_ss = k_ss;
params.l_ss = l_ss_target;

% Initial guess: a = [1, 0, ..., 0], b = [l_ss, 0, ..., 0]
a0 = zeros(n, 1);
a0(1) = 1; % Corresponds to phi_1(k) = k
b0 = zeros(n, 1);
b0(1) = l_ss_target; % Corresponds to psi_1(k) = 1
initial_guess = [a0; b0];

% Solve for coefficients using fsolve
options = optimoptions('fsolve', 'Display', 'iter', 'MaxFunctionEvaluations', 10000, 'MaxIterations', 1000);
ab_solution = fsolve(@(ab) model_residuals(ab, params), initial_guess, options);

% Extract coefficients
a_coeff = ab_solution(1:n);
b_coeff = ab_solution(n+1:end);

% Function to compute residuals
function residuals = model_residuals(ab, params)
    n = params.n;
    a_coeff = ab(1:n);
    b_coeff = ab(n+1:end);
    k_nodes = params.k_nodes;
    a_interval = params.a;
    b_interval = params.b;
    alpha = params.alpha;
    delta = params.delta;
    beta = params.beta;
    sigma = params.sigma;
    gamma = params.gamma;
    
    residuals = zeros(2*n, 1);
    
    for j = 1:n
        k_j = k_nodes(j);
        z_j = 2*(k_j - a_interval)/(b_interval - a_interval) - 1;
        
        % Basis functions for g(k_j) and h(k_j)
        phi = zeros(n, 1);
        psi = zeros(n, 1);
        for i = 1:n
            T_phi = cos((i-1) * acos(z_j));
            phi(i) = k_j * T_phi;
            psi(i) = T_phi;
        end
        
        k_prime_j = dot(a_coeff, phi);
        l_j = dot(b_coeff, psi);
        
        % Consumption at current node
        c_j = k_j^alpha * l_j^(1-alpha) + (1 - delta)*k_j - k_prime_j;
        
        % F_ℓ and F_k
        F_l_j = (1 - alpha) * k_j^alpha * l_j^(-alpha);
        F_k_j = alpha * k_j^(alpha - 1) * l_j^(1 - alpha);
        
        % Marginal utilities
        u_c_j = gamma * c_j^(gamma*(1 - sigma) - 1) * (1 - l_j)^((1 - gamma)*(1 - sigma));
        u_l_j = - (1 - gamma) * c_j^(gamma*(1 - sigma)) * (1 - l_j)^(-gamma - sigma*(1 - gamma));
        
        % Intra-temporal residual
        residuals(j) = u_l_j + u_c_j * F_l_j;
        
        % Evaluate next period's policy functions
        z_prime = 2*(k_prime_j - a_interval)/(b_interval - a_interval) - 1;
        if z_prime < -1 || z_prime > 1
            residuals(n + j) = 1e6; % Penalize out-of-bounds
            continue;
        end
        
        phi_prime = zeros(n, 1);
        psi_prime = zeros(n, 1);
        for i = 1:n
            T_phi_prime = cos((i-1) * acos(z_prime));
            phi_prime(i) = k_prime_j * T_phi_prime;
            psi_prime(i) = T_phi_prime;
        end
        
        k_prime_prime_j = dot(a_coeff, phi_prime);
        l_prime_j = dot(b_coeff, psi_prime);
        
        % Next period consumption
        c_prime_j = k_prime_j^alpha * l_prime_j^(1 - alpha) + (1 - delta)*k_prime_j - k_prime_prime_j;
        
        % Next period F_k
        F_k_prime = alpha * k_prime_j^(alpha - 1) * l_prime_j^(1 - alpha);
        
        % Euler equation residual
        u_c_prime = gamma * c_prime_j^(gamma*(1 - sigma) - 1) * (1 - l_prime_j)^((1 - gamma)*(1 - sigma));
        residuals(n + j) = u_c_j - beta * u_c_prime * (F_k_prime + 1 - delta);
    end
end

% --- After solving for ab_solution, extract coefficients ---
a_coeff = ab_solution(1:n);   % coefficients for capital policy
b_coeff = ab_solution(n+1:end); % coefficients for labor policy

% Define a grid over [a, b] for plotting
k_values = linspace(a, b, 100);
k_policy = zeros(size(k_values));
l_policy = zeros(size(k_values));
c_policy = zeros(size(k_values));  % optional: consumption policy

% Evaluate the policy functions on the grid
for idx = 1:length(k_values)
    k = k_values(idx);
    % Map k to the Chebyshev variable z in [-1,1]
    z = 2*(k - a)/(b - a) - 1;
    
    % Compute the Chebyshev basis functions for the current k:
    phi = zeros(n,1);
    psi = zeros(n,1);
    for i = 1:n
        T_i = cos((i-1) * acos(z));  % Chebyshev polynomial of order (i-1)
        phi(i) = k * T_i;            % Basis for the capital policy function
        psi(i) = T_i;                % Basis for the labor policy function
    end
    
    % Evaluate policy functions using the collocation coefficients:
    k_policy(idx) = dot(a_coeff, phi);
    l_policy(idx) = dot(b_coeff, psi);
    
    % Optional: Compute consumption using the model equation:
    % c = k^alpha * l^(1-alpha) + (1-delta)*k - k'
    c_policy(idx) = k^alpha * l_policy(idx)^(1-alpha) + (1-delta)*k - k_policy(idx);
end

% Plot the approximated policy functions
figure;

% Plot the next-period capital policy
subplot(3,1,1);
plot(k_values, k_policy, 'b-', 'LineWidth',2);
xlabel('Current Capital, k');
ylabel('Next Period Capital, k''');
title('Policy Function: Capital');
grid on;

% Plot the labor policy function
subplot(3,1,2);
plot(k_values, l_policy, 'r-', 'LineWidth',2);
xlabel('Current Capital, k');
ylabel('Labor, l');
title('Policy Function: Labor');
grid on;

% Plot the consumption policy function (if desired)
subplot(3,1,3);
plot(k_values, c_policy, 'm-', 'LineWidth',2);
xlabel('Current Capital, k');
ylabel('Consumption, c');
title('Policy Function: Consumption');
grid on;
