%% 3.b
clear all; close all; clc;

alpha = 0.36;
beta = 0.9;
N = 20;
sig = 0.01;

[lk, PPI] = tauchen(N, log(alpha*beta) / (1-alpha), alpha, sig, 3);

k = exp(lk);

PD = PPI - eye(N);
PD(:, N) = ones(N, 1);
b = zeros(N, 1);
b(N) = 1;

epr1 = linsolve(PD', b);

[V, D] = eig(PPI');
V = real(V);
D = real(D);

epr2 = V(:, 1)./(sum(V(:, 1)));

figure;
plot(k, epr1, '--b');
xlabel('capital');
ylabel('');
title('Ergodic Distribution');

% computing for different betas
%% 3.c - Expected Capital Computation
clear all; close all; clc;

alpha = 0.36;
sig = 0.01;
N = 20;
tauchen_std = 3; % Number of standard deviations for Tauchen grid

% Define the beta values
betas = [0.95, 0.97, 0.99];
expected_k_numerical = zeros(size(betas));
expected_k_analytical = zeros(size(betas));

% Loop over each beta
for i = 1:length(betas)
    beta = betas(i);
    
    % Compute the Tauchen discretization for log(k)
    [lk, PPI] = tauchen(N, log(alpha*beta)/(1-alpha), alpha, sig, tauchen_std);
    
    % Convert log capital grid back to levels
    k = exp(lk);
    
    % Compute the invariant distribution π using eigenvector method
    [V, D] = eig(PPI');
    
    % Find the eigenvector corresponding to the eigenvalue 1
    eigenvalues = diag(D);
    [~, index] = min(abs(eigenvalues - 1)); % Locate the eigenvalue closest to 1
    pi_invariant = V(:, index);
    
    % Normalize π to sum to 1
    pi_invariant = real(pi_invariant); % Remove numerical imaginary components
    pi_invariant = pi_invariant / sum(pi_invariant);
    
    % Compute the numerical expected capital: E[k] = sum(π(i) * k(i))
    expected_k_numerical(i) = log(sum(pi_invariant .* k));
    
    % Compute the analytical expected capital: exp(log(αβ) / (1 - α))
    expected_k_analytical(i) = log(alpha * beta) / (1 - alpha);
    
    % Plot the ergodic distribution
    figure;
    plot(k, pi_invariant, '--b', 'LineWidth', 2);
    xlabel('Capital (k)');
    ylabel('Invariant Distribution π');
    title(sprintf('Ergodic Distribution for \\beta = %.2f', beta));
    grid on;
end

% Display results with 4 decimal places (without the Difference column)
fprintf('Beta\tNumerical E[k]\tAnalytical E[k]\n');
for i = 1:length(betas)
    fprintf('%.2f\t%.4f\t%.4f\n', betas(i), expected_k_numerical(i), expected_k_analytical(i));
end
