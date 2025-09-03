clc; clear; close all;

alpha = 0.36;
beta  = 0.98;
sigma = 2;
% sigma = 1
delta = 0.03;
% delta = 1

% Compute a "steady-state" capital if you like:
kss = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));

% Production function and derivative
f  = @(k) k.^alpha;
fP = @(k) alpha * k.^(alpha - 1);

Nk     = 500;
kmin   = 0.001*kss;
kmax   = 1.5*kss;
kpGrid = linspace(kmin, kmax, Nk)';

u = @(c) c.^(1 - sigma)/(1 - sigma);

if abs(sigma - 1) < 1e-14
    u = @(c) log(c);
end

mu = @(c) c.^(-sigma);

Vk_guess = 1; 
cNext = ( beta * Vk_guess * ( fP(kpGrid) + 1 - delta ) ).^(-1/sigma);

maxIter = 1000;
tol     = 1e-8;
dist    = 1;
iter    = 0;

% Store policy on a "kGrid" = same as kpGrid
kGrid    = kpGrid;
cPolicy  = zeros(Nk,1);
kpPolicy = zeros(Nk,1);

while dist > tol && iter < maxIter
    iter = iter + 1;

    %Next‐period marginal utility
    muNext = mu(cNext);  

    eulerFactor = beta .* muNext .* ( fP(kpGrid) + 1 - delta );
    cToday      = eulerFactor.^(-1/sigma);

    kImplied = zeros(Nk,1);
    for j = 1:Nk
        target = cToday(j) + kpGrid(j);
        lo = 0; 
        hi = max(kmax, target); 
        for bIt = 1:50
            mid  = 0.5*(lo + hi);
            fVal = mid^alpha + (1-delta)*mid - target;
            if fVal > 0
                hi = mid;
            else
                lo = mid;
            end
        end
        kImplied(j) = 0.5*(lo + hi);
    end

    % Sort by kImplied, then interpolate:
    [kImp_sorted, idx] = sort(kImplied);
    cToday_sorted      = cToday(idx);
    kp_sorted          = kpGrid(idx);

    cPolicy_new  = interp1(kImp_sorted, cToday_sorted, kGrid, 'linear','extrap');
    kpPolicy_new = interp1(kImp_sorted, kp_sorted,     kGrid, 'linear','extrap');

    cPolicy_new(cPolicy_new < 1e-12) = 1e-12;

    % Update cNext
    cNext_new = interp1(kGrid, cPolicy_new, kpGrid, 'linear','extrap');
    cNext_new(cNext_new<1e-12) = 1e-12;

    % Check convergence
    diff_c  = max(abs(cPolicy_new  - cPolicy ));
    diff_kp = max(abs(kpPolicy_new - kpPolicy));
    dist    = max(diff_c, diff_kp);

    % Update
    cPolicy  = cPolicy_new;
    kpPolicy = kpPolicy_new;
    cNext    = cNext_new;
end

fprintf('EGM converged after %d iterations (dist=%.3e)\n', iter, dist);

% VALUE FUNCTION ITERATION
V_old = zeros(Nk,1);
critV = 1;
countV=0;
while critV>1e-10 && countV<1000
    countV = countV+1;
    V_kprime  = interp1(kGrid, V_old, kpPolicy, 'linear','extrap');
    V_new     = u(cPolicy) + beta*V_kprime;
    critV     = max(abs(V_new - V_old));
    V_old     = V_new;
end
V_final = V_old;
fprintf('Value function iteration took %d steps (crit=%.3e)\n', countV, critV);

% PLOTS
figure;
plot(kGrid, kpPolicy, 'LineWidth',2);
xlabel('k (today)'); ylabel('k''(k)');
title('Policy: Next-Period Capital');
grid on;

figure;
plot(kGrid, cPolicy, 'LineWidth',2);
xlabel('k (today)'); ylabel('c(k)');
title('Policy: Consumption');
grid on;

figure;
plot(kGrid, V_final, 'LineWidth',2);
xlabel('k (today)'); ylabel('V(k)');
title('Value Function');
grid on;
