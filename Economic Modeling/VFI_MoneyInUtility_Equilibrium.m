clear all;
clc;

N = 2;

e1 = 0.74;    % shock1
e2 = 1.36;    % shock2
theta = [e1 e2];

p11 = 0.73;
p12 = 0.27;
p21 = 0.27;
p22 = 0.73;
QQ = [p11 p12; p21 p22];

% Model parameters
bet = 0.98;  % Discount factor
sig = 2;     % Risk aversion
y = 1;       % Endowment
gama = 0.02; % Inflation rate

TT = 200;
msup = zeros(TT+1,1);  % Adjusted preallocation for msup
msup(1) = 1.4751;

for kk = 1:TT    
    % tau = 0.0234;
    % Initialize grids
    Np = 1000;      % Number of grid points, larger the point, closer to limit state
    Nk = 1000;   % Number of iterations    
    tau = msup(kk)*gama/(1+gama);   % Real money transfers
    m_low = y + tau;
    m_up = 2.5;  % Upper bound for money holdings
    mgrid = linspace(m_low, m_up, Np)';  % Define the grid for money holdings

    g0 = mgrid;
    g = zeros(Np, Nk, N);  % initial guess for policy on future money

    for i = 1:N
        g(:,1,i) = g0;
    end

    for j = 1:Nk-1 
        mc = zeros(Np, N);
        for i = 1:N
            mc(:,i) = theta(i) * (mgrid/(1+gama) + y + tau - g(:,j,i)).^(-sig);
        end
        vf = bet/(1+gama) * mc * QQ';
       
        ms = zeros(Np, N);
        for i = 1:N
            ms(:,i) = ((vf(:,i)/theta(i)).^(-1/sig) + mgrid - y - tau) * (1+gama);
        end
        for i = 1:Np
            for s = 1:N
                if mgrid(i) <= ms(1,s)
                    g(i,j+1,s) = mgrid(1);
                elseif mgrid(i) >= ms(Np,s)
                    g(i,j+1,s) = mgrid(Np);
                else
                    g(i,j+1,s) = interp1(ms(:,s), mgrid, mgrid(i), "linear");
                end
            end    
        end
        ac = max(abs(g(:,j+1,1) - g(:,j,1)), abs(g(:,j+1,2) - g(:,j,2)));
        if max(ac) <= 1e-5
            disp("convergence achieved");
            go = g(:,j+1,:);
            disp("number of iterations");
            disp(j);
            break;
        end
    end

    % Recovering policy
    gopt = zeros(Np, N);
    for i = 1:N
        gopt(:,i) = go(:,:,i);
    end
    
    Dp = 4 * Np;
    dgrid = linspace(m_low, m_up, Dp)';
    % dgrid = nugrid(dgrid,3);
    Fini = (dgrid - dgrid(1)) / (dgrid(Dp) - dgrid(1));
    
    %Fini = ones(Dp,1);
    Prob = QQ - eye(N);
    Prob(:,N) = ones(N,1);
    a = zeros(N,1);
    a(N) = 1;
    epr = linsolve(Prob', a);
    F = zeros(Dp, Np, N);
    for i = 1:N
        F(:,1,i) = epr(i) * Fini';
    end
    for i = 1:Nk-1
        for j = 1:Dp
            id = zeros(1, N);
            for s = 1:N
                if dgrid(j) < gopt(1,s)
                    id(1,s) = 0;
                elseif dgrid(j) > gopt(Np,s)
                    id(1,s) = epr(s);
                else
                    [goptr, index] = unique(gopt(:,s), "last");
                    id(1,s) = interp1(dgrid, F(:,i,s), interp1(goptr, mgrid(index), dgrid(j), "linear"), "linear");
                end
            end 
            for s = 1:N
                F(j,i+1,s) = id * QQ(:,s);
            end
        end
        h = max(abs(F(:,i+1,N) - F(:,i,N)));
        if h <= 1e-5
            FNN = F(:,i+1,:);
            disp("convergence of distributions achieved");
            disp(i);
            break;
        end
    end

    % Recovering distributions
    dopt = zeros(Dp, N);
    for i = 1:N
        dopt(:,i) = FNN(:,:,i);
    end

    FO = sum(dopt, 2);
    RB = (FO(2:Dp) - FO(1:Dp-1))' * ((dgrid(2:Dp) + dgrid(1:Dp-1))/2) + FO(1) * dgrid(1);
    
    msup(kk+1) = RB;
    
    if abs(msup(kk+1) - msup(kk)) <= 1e-5
        disp("real money supply found");
        disp(msup(kk));
        disp(kk+1);
        break;
    end
end

figure
plot(mgrid, gopt(:,1), 'b', 'LineWidth', 2);
hold on;
plot(mgrid, gopt(:,2), 'r', 'LineWidth', 2);
legend('Low Shock Policy', 'High Shock Policy');
xlabel('M');
ylabel('M\_prime');
title('Policy Function Convergence');
grid on;

figure
plot(dgrid, FNN(:,:,1), 'b', 'LineWidth', 2);
hold on;
plot(dgrid, FNN(:,:,2), 'r', 'LineWidth', 2);
legend('Low Shock Policy', 'High Shock Policy');
xlabel('M');
ylabel('F');
title('Distribution Convergence');
grid on;
