% Parameters
sigma = 2;
beta = 0.98;
y = 1;
gamma = 0.02;
tau = 0.0234;
theta = [0.74; 1.36]; % θ1 and θ2
Q = [0.73, 0.27; 0.27, 0.73]; % Transition matrix
m_min = y + tau; % Minimum m (1.0234)
m_max = 5; % Adjusted based on trial
n_m = 1000; % Grid points for m
n_theta = 2; % Two θ states

% Create grid for m
m_grid = linspace(m_min, m_max, n_m)';

% Initialize value and policy functions
V = zeros(n_theta, n_m); % Value function
policy_m = zeros(n_theta, n_m); % Policy for m'
policy_c = zeros(n_theta, n_m); % Policy for c

% VFI parameters
tolerance = 1e-6;
max_iter = 1000;
iter = 0;
diff = inf;

while diff > tolerance && iter < max_iter
    V_new = V;
    for theta_idx = 1:n_theta
        for m_idx = 1:n_m
            m_current = m_grid(m_idx);
            c_max = m_current / (1 + gamma);
            
            % Calculate c for all possible m'
            m_prime = m_grid;
            c = (m_current / (1 + gamma)) + y + tau - m_prime;
            
            % Ensure c > 0 and valid
            valid = c > 0;
            utility = theta(theta_idx) * (c.^(1 - sigma)) / (1 - sigma);
            utility(~valid) = -inf;
            
            % Compute expected future value
            EV = zeros(n_m, 1);
            for theta_prime = 1:n_theta
                EV = EV + Q(theta_idx, theta_prime) * interp1(m_grid, V(theta_prime, :), m_prime, 'linear', -inf);
            end
            total_value = utility + beta * EV;
            
            % Find optimal m'
            [max_val, max_idx] = max(total_value);
            V_new(theta_idx, m_idx) = max_val;
            policy_m(theta_idx, m_idx) = m_prime(max_idx);
            policy_c(theta_idx, m_idx) = c(max_idx);
        end
    end
    diff = max(abs(V_new(:) - V(:)));
    V = V_new;
    iter = iter + 1;
end

% Plot combined policy functions
figure;

% Consumption policies
subplot(2,1,1);
plot(m_grid, policy_c(1, :), 'r-', m_grid, policy_c(2, :), 'b--', 'LineWidth', 2);
title('Consumption Policy Functions');
xlabel('m');
ylabel('c(m, θ)');
legend('θ₁ = 0.74', 'θ₂ = 1.36', 'Location', 'best');
grid on;

% Savings policies
subplot(2,1,2);
plot(m_grid, policy_m(1, :), 'r-', m_grid, policy_m(2, :), 'b--', 'LineWidth', 2);
title('Savings Policy Functions');
xlabel('m');
ylabel('m''(m, θ)');
legend('θ₁ = 0.74', 'θ₂ = 1.36', 'Location', 'best');
grid on;