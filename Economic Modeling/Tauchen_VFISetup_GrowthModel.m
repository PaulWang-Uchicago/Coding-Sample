%% Initialization and Timing
clc;            % Clear the command window
clear all;      % Remove all existing variables
ts = tic;       % Start a timer for performance measurement

%% Parameter Setup
% Define the number of states for the Markov process and numerical parameters for the grid and iterations.
N = 5;              % Number of discrete states for the shock process
Np = 1000;          % Number of grid points for the capital grid
Nk = 1000;          % Maximum iterations for the value function loop

% Model parameters
alpha = 0.36;       % Capital's share in production
beta = 0.98;        % Discount factor for future utility
sigma = 0.1;        % Curvature parameter of the utility function (risk aversion)
delta = 0.03;       % Rate of depreciation for capital
rho = 0.70;         % Persistence parameter of the shock process
gamma = 2;          % Risk aversion parameter (used in utility)
sigmaz = sqrt(sigma^2/(1-rho^2));  % Standard deviation of the shock process, adjusted for persistence

% Compute steady-state capital level based on the model parameters.
kss = (alpha * beta / (1 - (1 - delta) * beta))^(1 / (1 - alpha));

% Generate a capital grid ranging from 1% of steady state to 4 times steady state.
kgrid = linspace(0.01 * kss, 4 * kss, Np)';

%% Part (a): Discretize the Shock Process and Find Its Ergodic Distribution

% Use the Tauchen method to approximate the AR(1) process for shocks.
% 'a' will be the discrete grid for ln(z) and 'PP' is the transition probability matrix.
[a, PP] = tauchen(N, 0, rho, sigmaz, 3);

% Compute the ergodic (stationary) distribution for the Markov chain.
% This is done by finding the eigenvector of the transposed transition matrix corresponding to eigenvalue 1.
eigvals = eig(PP');          % Compute eigenvalues (not used further explicitly)
[V, D] = eig(PP');           % Get eigenvectors and eigenvalues
[~, idx] = min(abs(diag(D) - 1));  % Identify the eigenvalue closest to 1
ergodic_dist = V(:, idx) / sum(V(:, idx));  % Normalize the eigenvector so the elements sum to one

% Display the state grid, transition matrix, and the stationary distribution.
disp('State space (ln z_t):');
disp(a');
disp('Transition Probability Matrix:');
disp(PP);
disp('Ergodic Distribution:');
disp(ergodic_dist');

% Plot a bar chart of the ergodic distribution (after converting ln(z) to z by exponentiation).
figure;
plot(exp(a), ergodic_dist, 'b');
xlabel('z_t (Productivity Shock)');
ylabel('Probability');
title('Ergodic Distribution of z_t');
grid on;

%% Part (b): Compute Consumption, Returns, and Solve the Value Function

% Convert the shock grid from logarithms to levels.
z = exp(a);

% For each shock state, calculate consumption given production and investment.
% The consumption function is: consumption = output (z * k^alpha) + remaining capital (1-delta)*k - new capital (k').
% The code computes this for each combination of current capital (kgrid) and next period capital (also kgrid).
consu1 = z(1) * kgrid.^alpha + (1 - delta) * kgrid - kgrid';
consu2 = z(2) * kgrid.^alpha + (1 - delta) * kgrid - kgrid';
consu3 = z(3) * kgrid.^alpha + (1 - delta) * kgrid - kgrid';
consu4 = z(4) * kgrid.^alpha + (1 - delta) * kgrid - kgrid';
consu5 = z(5) * kgrid.^alpha + (1 - delta) * kgrid - kgrid';

% Replace any non-positive consumption values with zero.
consu1(consu1 <= 0) = 0;
consu2(consu2 <= 0) = 0;
consu3(consu3 <= 0) = 0;
consu4(consu4 <= 0) = 0;
consu5(consu5 <= 0) = 0;

% Calculate the instantaneous returns for each state using a utility function 'ut'.
Ret1 = ut(gamma, consu1);
Ret2 = ut(gamma, consu2);
Ret3 = ut(gamma, consu3);
Ret4 = ut(gamma, consu4);
Ret5 = ut(gamma, consu5);

% Initialize matrices to store the value functions for each shock state.
v1 = zeros(Np, Nk);
v2 = zeros(Np, Nk);
v3 = zeros(Np, Nk);
v4 = zeros(Np, Nk);
v5 = zeros(Np, Nk);

% Set the initial guess for the value function using utility evaluated at the capital grid.
v1(:,1) = ut(gamma, kgrid);
v2(:,1) = ut(gamma, kgrid);
v3(:,1) = ut(gamma, kgrid);
v4(:,1) = ut(gamma, kgrid);
v5(:,1) = ut(gamma, kgrid);

% Start the value function iteration.
for i = 1:Nk-1
    % For each state, update the value function by maximizing over the choices for next period's capital.
    % The update involves the instantaneous return and the discounted expected future value.
    v1(:,i+1) = max((Ret1 + beta * PP(1,1) * v1(:,i)' + beta * PP(1,2) * v2(:,i)' + beta * PP(1,3) * v3(:,i)' + beta * PP(1,4) * v4(:,i)' + beta * PP(1,5) * v5(:,i)')')';
    v2(:,i+1) = max((Ret2 + beta * PP(2,1) * v1(:,i)' + beta * PP(2,2) * v2(:,i)' + beta * PP(2,3) * v3(:,i)' + beta * PP(2,4) * v4(:,i)' + beta * PP(2,5) * v5(:,i)')')';
    v3(:,i+1) = max((Ret3 + beta * PP(3,1) * v1(:,i)' + beta * PP(3,2) * v2(:,i)' + beta * PP(3,3) * v3(:,i)' + beta * PP(3,4) * v4(:,i)' + beta * PP(3,5) * v5(:,i)')')';
    v4(:,i+1) = max((Ret4 + beta * PP(4,1) * v1(:,i)' + beta * PP(4,2) * v2(:,i)' + beta * PP(4,3) * v3(:,i)' + beta * PP(4,4) * v4(:,i)' + beta * PP(4,5) * v5(:,i)')')';
    v5(:,i+1) = max((Ret5 + beta * PP(5,1) * v1(:,i)' + beta * PP(5,2) * v2(:,i)' + beta * PP(5,3) * v3(:,i)' + beta * PP(5,4) * v4(:,i)' + beta * PP(5,5) * v5(:,i)')')';
    
    % Check the convergence by computing the difference between successive iterations.
    v_diff = [v1(:,i+1); v2(:,i+1); v3(:,i+1); v4(:,i+1); v5(:,i+1)] - [v1(:,i); v2(:,i); v3(:,i); v4(:,i); v5(:,i)];
    if max(abs(v_diff)) <= 0.001 * (1 - beta)
        % When the change is small enough, store the indices of the optimal policy.
        [v1s, kindex1] = max((Ret1 + beta * PP(1,1) * v1(:,i)' + beta * PP(1,2) * v2(:,i)' + beta * PP(1,3) * v3(:,i)' + beta * PP(1,4) * v4(:,i)' + beta * PP(1,5) * v5(:,i)')');
        [v2s, kindex2] = max((Ret2 + beta * PP(2,1) * v1(:,i)' + beta * PP(2,2) * v2(:,i)' + beta * PP(2,3) * v3(:,i)' + beta * PP(2,4) * v4(:,i)' + beta * PP(2,5) * v5(:,i)')');
        [v3s, kindex3] = max((Ret3 + beta * PP(3,1) * v1(:,i)' + beta * PP(3,2) * v2(:,i)' + beta * PP(3,3) * v3(:,i)' + beta * PP(3,4) * v4(:,i)' + beta * PP(3,5) * v5(:,i)')');
        [v4s, kindex4] = max((Ret4 + beta * PP(4,1) * v1(:,i)' + beta * PP(4,2) * v2(:,i)' + beta * PP(4,3) * v3(:,i)' + beta * PP(4,4) * v4(:,i)' + beta * PP(4,5) * v5(:,i)')');
        [v5s, kindex5] = max((Ret5 + beta * PP(5,1) * v1(:,i)' + beta * PP(5,2) * v2(:,i)' + beta * PP(5,3) * v3(:,i)' + beta * PP(5,4) * v4(:,i)' + beta * PP(5,5) * v5(:,i)')');
  
        disp("convergence achieved");
        break;
    end
end

% Collect the optimal policy indices for each state.
kindex = [kindex1' kindex2' kindex3' kindex4' kindex5'];
kpol = zeros(Np, 5);

% Map the indices back to the actual capital grid values.
for j = 1:5
    for i = 1:Np
        kpol(i,j) = kgrid(kindex(i,j));
    end
end

% Gather the converged value function values for each shock state.
values = [v1s' v2s' v3s' v4s' v5s'];

% Plot the value functions for each shock state against the capital grid.
figure;
plot(kgrid, values(:,1), 'b', kgrid, values(:,2), 'r', kgrid, values(:,3), 'g', kgrid, values(:,4), 'k', kgrid, values(:,5), 'm');
legend('v1', 'v2', 'v3', 'v4', 'v5');
xlabel('Capital Grid');
ylabel('Value Function');
title('Value Functions against Capital Grid');
grid on;

% Plot the policy functions (optimal capital choices) for each shock state.
figure;
plot(kgrid, kpol(:,1), 'b', kgrid, kpol(:,2), 'r', kgrid, kpol(:,3), 'g', kgrid, kpol(:,4), 'k', kgrid, kpol(:,5), 'm');
legend('kpol1', 'kpol2', 'kpol3', 'kpol4', 'kpol5');
xlabel('Capital Grid');
ylabel('Policy Function');
title('Policy Function Convergence');
grid on;

%% Utility Function Definition
% Define the utility function, which handles both the CRRA (if gamma ≠ 1) and log cases.
function c = ut(g, b)
    if g ~= 1
        c = (1 / (1 - g)) * b.^(1 - g);
    else
        c = log(b);
    end
end

%% Part (c): Simulating the Markov Chain and Capital Dynamics

% Set the simulation length.
TT = 50000;  % Total number of periods to simulate

% Initialize the state sequence for the Markov chain.
x = zeros(TT,1);
x(1) = 5;  % Starting state (arbitrarily chosen as state 5)

% Simulate the Markov chain over TT periods.
for i = 1:TT-1
    % Use cumulative probabilities to determine the next state.
    x(i+1) = find(rand <= cumsum(PP(x(i),:)), 1);
end

% Initialize and simulate the evolution of capital.
simk = zeros(TT,1);
simk(1) = kgrid(100);  % Start from a specific point on the capital grid

for i = 1:TT-1
    % For the current capital value, find its index in the grid and use the policy function corresponding to the next state.
    simk(i+1) = kpol(find(kgrid == simk(i)), x(i+1));  
end

% Plot the simulated capital path over time.
figure;
plot(1:TT, simk);
grid;
xlabel('Time');
ylabel('Capital');
title('Simulated Capital Dynamics');
legend({'g1','g2','g3','g4','g5'}, 'Location', 'northeast');

% Compute the average capital after a burn-in period.
avk = mean(simk(1000:TT));

% Plot a histogram showing the ergodic distribution of capital.
figure;
histogram(simk, 'Normalization', 'probability');
grid on;
xlabel('Capital');
ylabel('Frequency');
title('Ergodic Distribution of Capital');

%% Additional Helper Function: Discrete Inverse Random Sampling
% This function returns random draws from a discrete distribution specified by probabilities 'p'
function X = discreteinvrnd(p, m, n)
    X = zeros(m, n);  % Preallocate an output matrix
    for i = 1:m*n
        u = rand;
        I = find(u < cumsum(p));
        X(i) = min(I);
    end
end
