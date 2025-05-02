yalmip('clear');
clear;
close all;

rng(100); % For reproducibility

%% System and Model Parameters
C = [0 0 1 0; 0 0 0 1];
nx = 4; nu = 2; ny = 2;
dt = 1; cv = 5.435e-5;
rhos = 1037; rhoc = 1424; qs = 920; g = 9.81;

% Pressure head model coefficients
p1 = 0.2242; p2 = 7.95e-5; 
p3 = -3.9985e-4; p4 = -280.29;

% Input bounds
umin = [0; 2877]; umax = [1; 3356];

% Degradation and hazard model parameters
lamL = 6e-7; lam0 = 3e-6; zeta = 0.8; 
T = 26280;  N = 3; tau = round(T / N); 
tD = round(T * rand()); 

% Health-aware model parameters
uHR = 1.8e-8; gamma = 4e-5; b0 = 5; alpha = 0.5; beta = 0.3; lamf = 4e-7;

% MPC configuration
Q = eye(2); Rd = 0.5 * eye(2); Rh = 0.5;
Np = 2; Nr = 1;

%% Scenario Tree Setup
dval_list = {[-0.05, 0.05], [-0.05, 0.05]};
tree = DistTree(dval_list, Nr);
Ns = size(tree, 1); 

%% YALMIP Variables
u = sdpvar(nu, Np, Ns, 'full');
x = sdpvar(nx, Np + 1, Ns, 'full');
r = sdpvar(ny, Np + 1, Ns, 'full');
pfd = sdpvar(1, Np, Ns, 'full');
pastu = sdpvar(nu, 1, 'full');

%% MPC Objective and Constraints
objective = 0;
constraints = [];

for s = 1:Ns
    du = diff([pastu, u(:,:,s)], 1, 2);
    for k = 1:Np
        dp = squeeze(tree(s,:,min(k,Nr)));
        nu_norm = (u(:,k,s) - umin) ./ (umax - umin);
        hr = pfd(:,k,s) * (lamL + lam0 * exp(zeta * nu_norm(2)));

        % Output tracking and health-aware penalty
        objective = objective + ...
            ((C*x(:,k,s) - r(:,k,s))' * Q * (C*x(:,k,s) - r(:,k,s)) + ...
             Rh * hr + du(:,k)' * Rd * du(:,k)) / Ns;

        % System dynamics
        h = p1*u(2,k,s) + p2*u(2,k,s)*(qs + x(3,k,s)) + ...
            p3*(qs + x(3,k,s))^2 + p4;

        % Dynamics constraints
        constraints = [constraints, ...
            x(:,k+1,s) == [ ...
                x(1,k,s) + dp(1)*dt;
                x(2,k,s) + dp(2)*dt;
                cv * u(1,k,s) * ((x(2,k,s) - x(1,k,s)) * 1e5 / rhos)^0.5;
                x(1,k,s) + rhos * g * h * 1e-5]];
        
        % Input and health constraint
        constraints = [constraints, ...
            umin <= u(:,k,s) <= umax, ...
            hr <= uHR];
    end
end

%% Non-anticipativity Constraints
for i = 1:Nr
    for s1 = 1:Ns
        for s2 = s1+1:Ns
            if isequal(tree(s1,:,1:i), tree(s2,:,1:i))
                constraints = [constraints, u(:,i,s1) == u(:,i,s2)];
            end
        end
    end
end

%% Controller Setup
parameters_in = {x(:,1,:), r(:,:,:), pastu, pfd(:,:,:)};
solutions_out = {u(:,1,1)};
controller = optimizer(constraints, objective, ...
    sdpsettings('solver', 'fmincon'), parameters_in, solutions_out);

%% Initial Conditions
xk = [200.2; 216.2; 0; 0]; % Initial state
oldu = [0.8; 3063];
xhist = xk; uhist = oldu; nuhist = [];
P0 = [0 0 0 1]; P = P0;
PFD = P(1); PFDCum = PFD;
HRCum = 0; t = 0; n = 0; m = 0; iDx = 0;

%% Simulation Loop
hbar = waitbar(0, "Simulation Progression...");
for j = 1:(T/dt)+1
    tic;

    % Predict future PFD
    PM = P; nM = n; mM = m; iDxM = iDx;
    for l = 1:Np
        bM = b0 * exp(-alpha * nM - beta * mM);
        lamM = gamma * (1 - (1 - exp(-bM)));
        AM = [0 0 0 0;
             lamf+lamM -(lamf+lamM) 0 0;
             lamf lamM -(lamf+lamM) 0;
             lamf 0 lamM -(lamf+lamM)];
        PM = PM * (eye(4) + AM * dt);

        % Training event
        if mod((j+l)*dt, tau) == 0 && (j+l)*dt ~= 0
            if iDxM == 0
                mM = mM + 1;
                PM = [0 PM(2) PM(3) PM(4) + PM(1)];
            else
                iDxM = 0;
            end
        end

        % Degradation event
        if (j+l)*dt == tD
            nM = nM + 1;
            PM = [0 PM(2) PM(3) PM(4) + PM(1)];
            iDxM = 1;
        end
        PFDM(l) = PM(1);
    end

    futurePFD = repmat(PFDM, 1, 1, Ns);
    rk = [1.708e-3 * ones(1, Np+1, Ns); 230.2 * ones(1, Np+1, Ns)];
    inputs = {repmat(xk, 1, 1, Ns), rk, oldu, futurePFD};
    [uk, diagnostics] = controller{inputs};

    if diagnostics == 1
        disp('Infeasible MPC problem.');
        break;
    end

    % Process & measurement noise
    vk = [-0.001 + 0.002 * sum(rand(1,100),2) / 100; ...
        -0.001 + 0.002 * sum(rand(1,100),2) / 100; ...
        -0.5e-3 + 1e-3 * sum(rand(1,100),2) / 100; ...
        -0.001 + 0.002 * sum(rand(1,100),2) / 100]; 
    wk = [-0.5e-3 + 1e-3 * sum(rand(1,100),2) / 100; ...
        -0.001 + 0.002 * sum(rand(1,100),2) / 100];

    % System update
    dpk = [-0.05 + 0.1*rand(); -0.05 + 0.1*rand()];
    rhok = (qs*rhos^2 + xk(3)*rhoc^2) / (qs*rhos + xk(3)*rhoc);
    hk = p1*uk(2) + p2*uk(2)*(qs + xk(3)) + p3*(qs + xk(3))^2 + p4;
    xk = [xk(1) + dpk(1)*dt;
          xk(2) + dpk(2)*dt;
          cv * uk(1) * sqrt((xk(2) - xk(1)) * 1e5 / rhok);
          xk(1) + rhok * g * hk * 1e-5] + vk * dt;
    yk = C * xk + wk;

    % Update degradation model
    b = b0 * exp(-alpha * n - beta * m);
    lam = gamma * (1 - (1 - exp(-b)));
    A = [0 0 0 0;
         lamf+lam -(lamf+lam) 0 0;
         lamf lam -(lamf+lam) 0;
         lamf 0 lam -(lamf+lam)];
    P = P * (eye(4) + A * dt);

    % Training and degradation
    if mod(t, tau) == 0 && t ~= 0
        if iDx == 0
            m = m + 1;
            P = [0 P(2) P(3) P(4) + P(1)];
            xk = [200.2; 216.2; 0; 0]; uk = [0.8; 3063];
        else
            iDx = 0;
        end
    end
    if t == tD
        n = n + 1;
        P = [0 P(2) P(3) P(4) + P(1)];
        xk = [200.2; 216.2; 0; 0]; uk = [0.8; 3063];
        iDx = 1;
    end

    % Compute health metrics
    PFD(j+1) = P(1);
    nuk = (uk - umin) ./ (umax - umin);
    lamP = lam0 * exp(zeta * nuk(2));
    DR(j) = lamL + lamP;
    HR(j) = PFD(j+1) * DR(j);

    % Tracking errors
    eq(j) = rk(1,1,1) - xk(3);
    ep(j) = rk(2,1,1) - xk(4);

    % Log history
    PFDCum = PFDCum + PFD(j + 1);
    HRCum = HRCum + HR(j);
    oldu = uk;
    xhist = [xhist xk];
    uhist = [uhist uk];
    nuhist = [nuhist nuk];
    t = t + dt;
    waitbar(t / T, hbar);
    delta(j) = toc;
end
close(hbar);

%% Evaluation
rq = sqrt(mean(eq.^2));
rp = sqrt(mean(ep.^2));
PFDAvg = PFDCum / (T/dt + 2);
HRAvg = HRCum / (T/dt + 1);

times = linspace(0, T, j);

%% State Tracking Plot
fh1 = figure(1); clf;
tiledlayout(2,1, 'Padding', 'compact', 'TileSpacing', 'compact');

% Chlorine Flow
nexttile
plot(times, xhist(3,1:end-1), 'LineWidth', 2, 'Color', [0 0.45 0.74]); hold on
yline(1.708e-3, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.85 0.33 0.1]);
xlim([0 times(end)]); ylim([0 0.4e-2])
xticks([0 tau 2*tau 3*tau]); xticklabels({'0','\tau','2\tau','3\tau'})
legend({'$q_c\;(t)$','$q_c^{ref}$'}, 'Interpreter', 'latex', 'Location', 'best')
ylabel('$q_c\;(m^3h^{-1})$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

% Feed Pressure
nexttile
plot(times, xhist(4,1:end-1), 'LineWidth', 2, 'Color', [0.47 0.67 0.19]); hold on
yline(230.2, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.85 0.33 0.1]);
xlim([0 times(end)]); ylim([210 240])
xticks([0 tau 2*tau 3*tau]); xticklabels({'0','\tau','2\tau','3\tau'})
legend({'$p_f\;(t)$','$p_f^{ref}$'}, 'Interpreter', 'latex', 'Location', 'best')
ylabel('$p_f\;(bar)$', 'Interpreter','latex')
xlabel('$t\;(s)$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

%% Control & Risk Metrics Plot
fh2 = figure(2); clf;
tiledlayout(3,1, 'Padding', 'compact', 'TileSpacing', 'compact');

% PFD
nexttile
plot(times, PFD(1:end-1), 'LineWidth', 2, 'Color', [0.49 0.18 0.56]); hold on
yline(PFDAvg, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.85 0.33 0.1]);
xlim([0 times(end)])
xticks([0 tau 2*tau 3*tau]); xticklabels({'0','\tau','2\tau','3\tau'})
legend({'$PFD\;(t)$','$PFD$'}, 'Interpreter', 'latex', 'Location', 'best')
ylabel('$PFD$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

% Control Inputs
nexttile
plot(times, nuhist(1,:), 'LineWidth', 2, 'Color', [0.93 0.69 0.13]); hold on
plot(times, nuhist(2,:), 'LineWidth', 2, 'Color', [0.3 0.75 0.93]);
xlim([0 times(end)]); ylim([0 1]);
xticks([0 tau 2*tau 3*tau]); xticklabels({'0','\tau','2\tau','3\tau'})
legend({'$z^*\;(t)$','$\omega^*\;(t)$'}, 'Interpreter', 'latex', 'Location', 'best')
ylabel('$u^*$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

% Hazard Rate
nexttile
plot(times, HR, 'LineWidth', 2, 'Color', [0.64 0.08 0.18]); hold on
yline(HRAvg, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0 0.45 0.74]);
yline(uHR, 'LineWidth', 2, 'LineStyle', ':', 'Color', [0.47 0.67 0.19]);
xlim([0 times(end)]); ylim([0 2.2e-8])
xticks([0 tau 2*tau 3*tau]); xticklabels({'0','\tau','2\tau','3\tau'})
legend({'$HR\;(t)$','$HR$','$\overline{HR}$'}, 'Interpreter','latex', 'Location','best')
ylabel('$HR$', 'Interpreter','latex')
xlabel('$t\;(s)$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)
xlabel('$t\;(s)$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)
