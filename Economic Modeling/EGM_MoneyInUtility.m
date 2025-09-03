%% Clear Workspace and Command Window
clear; clc;

%% Transition Matrix and Shocks Setup
% Transition probabilities
p11 = 0.73; p12 = 0.27;
p21 = 0.27; p22 = 0.73;
QQ = [p11 p12; p21 p22];
ns = 2;

% Shocks for two states
e1 = 0.74;   % Shock 1
e2 = 1.36;   % Shock 2
theta = [e1 e2];

%% Model Parameters
bet  = 0.98;    % Discount factor
sig  = 2;       % Risk aversion coefficient
y    = 1;       % Endowment
gama = 0.02;    % Inflation rate
msup = 1;       % Money supply (not used further in code)
tau  = 0.0234;  % Real money transfer

Nk = 5000;      % Maximum iterations for policy function convergence

%% Define Money Grid
Np = 1000;
lower_grid = y + tau;
upper_grid = 2.5;
mgrid = linspace(lower_grid, upper_grid, Np)';

%% Initial Guess for Endogenous Grid Method
g0 = mgrid;              % Initial guess for future money policy
g = zeros(Np, Nk, ns);   % Policy function array

% Set initial policy for both shocks
for s = 1:ns
    g(:, 1, s) = g0;
end

%% Policy Function Iteration
for iter = 1:Nk-1
    % Compute marginal consumption based on budget constraint for each shock
    mc = zeros(Np, ns);
    for s = 1:ns
        mc(:, s) = theta(s) * ( mgrid./(1+gama) + y + tau - g(:, iter, s) ).^(-sig);
    end
    
    % Value function calculation using the envelope condition
    vf = bet/(1+gama) * mc * QQ';
    
    % Update money supply policy using first-order conditions
    ms_policy = zeros(Np, ns);
    for s = 1:ns
        ms_policy(:, s) = ( (vf(:, s)/theta(s)).^(-1/sig) + mgrid - y - tau ) * (1+gama);
    end

    % Update policy function by interpolation with respect to the budget constraint
    for i = 1:Np
        for s = 1:ns
            if mgrid(i) <= ms_policy(1, s)
                g(i, iter+1, s) = mgrid(1);
            elseif mgrid(i) >= ms_policy(end, s)
                g(i, iter+1, s) = mgrid(end);
            else
                g(i, iter+1, s) = interp1(ms_policy(:, s), mgrid, mgrid(i));
            end
        end
    end
    
    % Convergence check using the policy for the last shock
    diffPolicy = abs(g(:, iter+1, ns) - g(:, iter, ns));
    if max(diffPolicy) <= 1e-10
        disp('Convergence achieved for policy function.');
        disp(['Number of iterations: ', num2str(iter)]);
        g_converged = g(:, iter+1, :);
        break;
    end
end

% Extract the converged policy function for each shock
gopt = zeros(Np, ns);
for s = 1:ns
    gopt(:, s) = g_converged(:, :, s);
end

%% Compute the Stationary Distribution
% Extract policy functions for each shock state
g1 = gopt(:, 1);
g2 = gopt(:, 2);

% Define a finer grid for the distribution
Dp = 2*Np - 1;
dgrid = linspace(lower_grid, upper_grid, Dp)';

% Initial cumulative distribution (linear)
Fini = (dgrid - dgrid(1)) / (dgrid(end) - dgrid(1));

% Compute invariant probabilities from QQ
Prob = QQ - eye(ns);
Prob(:, ns) = ones(ns, 1);
a = zeros(ns, 1);
a(ns) = 1;
epr = linsolve(Prob', a);

% Set up iteration for the distribution
Dk = 1000;
F1 = zeros(Dp, Dk);
F2 = zeros(Dp, Dk);
F1(:, 1) = epr(1) * Fini;
F2(:, 1) = epr(2) * Fini;

% Iterate to compute distributions for shock 1 and shock 2
for iter = 1:Dk-1
    for j = 1:Dp
        % Shock 1: Interpolate distribution value
        if dgrid(j) < g1(1)
            m1 = 0;
        elseif dgrid(j) > g1(end)
            m1 = epr(1);
        else
            [g1_unique, idx1] = unique(g1, "last");
            m1 = interp1(dgrid, F1(:, iter), interp1(g1_unique, mgrid(idx1), dgrid(j), "linear"));
        end
        
        % Shock 2: Interpolate distribution value
        if dgrid(j) < g2(1)
            m2 = 0;
        elseif dgrid(j) > g2(end)
            m2 = epr(2);
        else
            [g2_unique, idx2] = unique(g2, "last");
            m2 = interp1(dgrid, F2(:, iter), interp1(g2_unique, mgrid(idx2), dgrid(j), "linear"), "linear");
        end
        
        % Update distributions using transition probabilities
        F1(j, iter+1) = m1*QQ(1,1) + m2*QQ(2,1);
        F2(j, iter+1) = m1*QQ(1,2) + m2*QQ(2,2);
    end
    
    % Check convergence for both distributions
    v = [F1(:, iter+1); F2(:, iter+1)] - [F1(:, iter); F2(:, iter)];
    if max(abs(v)) <= 1e-10
        F1N = F1(:, iter+1);
        F2N = F2(:, iter+1);
        disp('Convergence achieved for distributions.');
        break;
    end
end

%% Plotting Results

% Shock 1 - Policy Function
subplot(2,2,1);
plot(mgrid, gopt, 'r-', 'LineWidth', 1.5);
hold on;
plot(mgrid, mgrid, 'b--', 'LineWidth', 1);
hold off;
title('Policy Function for M (Shock 1)');
ylabel('Next Period M');
grid on;

% Shock 1 - Distribution
subplot(2,2,3);
plot(dgrid, F1N, 'r-', 'LineWidth', 1.5);
title('Distribution (Shock 1)');
xlabel('M');
ylabel('Shock 1 Distribution');
grid on;

% Shock 2 - Policy Function
subplot(2,2,2);
plot(mgrid, gopt, 'r-', 'LineWidth', 1.5);
hold on;
plot(mgrid, mgrid, 'b--', 'LineWidth', 1);
hold off;
title('Policy Function for M (Shock 2)');
ylabel('Next Period M');
grid on;

% Shock 2 - Distribution
subplot(2,2,4);
plot(dgrid, F2N, 'r-', 'LineWidth', 1.5);
title('Distribution (Shock 2)');
xlabel('M');
ylabel('Shock 2 Distribution');
grid on;
