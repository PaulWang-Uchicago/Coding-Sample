function [c_policy, m_prime] = solve_model_EGM(tau, m_grid, theta_values, Q, sigma, beta, y, gamma)    
    num_m = length(m_grid);
    num_theta = length(theta_values);
    inv_1_plus_gamma = 1 / (1 + gamma);
    
    % Initialize policy functions
    c_policy = zeros(num_m, num_theta);
    m_prime = zeros(num_m, num_theta);
    
    % Precompute griddedInterpolant for faster interpolation
    F = griddedInterpolant(m_grid, m_grid, 'linear', 'nearest');
    
    % Parallelize over theta (remove 'parfor' if no Parallel Toolbox)
    parfor j = 1:num_theta
        theta = theta_values(j);
        c_current = zeros(num_m, 1);
        m_current = zeros(num_m, 1);
        
        % EGM steps
        for i = 1:num_m
            % Future marginal utility (corrected Euler equation)
            mu_future = sum(theta_values .* (beta * Q(j, :)) .* (F(m_grid(i))).^(-sigma));
            c_current(i) = (mu_future)^(-1/sigma);
            
            % Budget constraint
            m_current(i) = (m_grid(i) + c_current(i) - y - tau) * (1 + gamma);
        end
        
        % Interpolate back to the grid
        valid = m_current >= m_grid(1) & m_current <= m_grid(end);
        c_policy(:, j) = interp1(m_current(valid), c_current(valid), m_grid, 'linear', 0);
        m_prime(:, j) = interp1(m_current(valid), m_grid(valid), m_grid, 'linear', 0);
    end
end