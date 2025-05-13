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
T = 73000; tau0 = 9192.57; eta = 0.8922;
% REDUCE T FOR FASTER SIMULATION

% Health-aware model parameters
uHR = 1.5e-8; gamma = 7e-5; b0_2 = 100; 
b0_1 = 75; alpha = 0.3; lamf = 4e-7;

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
HRCum = 0; t = 0; n = 0; o = 4; N = 16; iDx = 0;
P0 = zeros(1,N+o); P0(1) = 1; P = P0;
PFD = P(2); PFDCum = PFD; tau_c = tau0; tau = tau0;

% Degradation model
A = zeros(N+o); A(1,2) = lamf;
A(1,o+n) = lmbda(gamma, b0_2, alpha, n);
A(o+n,2) = lmbda(gamma, b0_1, alpha, n) + lamf;
A = fixA(A);

%% Simulation Loop
hbar = waitbar(0, "Simulation Progression...");
for j = 1:(T/dt)+1
    tic;

    % Predict future PFD
    PM = P; nM = n; AM = A; tauM = tau; tau_cM = tau_c;
    for l = 1:Np
        PM = PM * (eye(N+o) + AM * dt);

        % Training event
        if mod((j+l)*dt, ceil(tau_cM)) == 0 && (j+l)*dt ~= 0
            nM = nM + 1;
            PM(3) = PM(3) + PM(2); PM(2) = 0;  
            AM(1,o+nM) = lmbda(gamma, b0_2, alpha, nM);
            A(1,o+nM-1) = 0;
            for i = 0:nM
                AM(o+i,2) = lmbda(gamma, b0_1, alpha, nM-i);
            end
            AM = fixA(AM);
            tauM = eta * tauM;
            tau_cM = tau_cM + tauM;
        end

        PFDM(l) = PM(2);
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

    % Update degradation
    P = P * (eye(N+o) + A * dt);

    % Testing and degradation
    if mod(t, ceil(tau_c)) == 0 && t ~= 0
        n = n + 1;
        P(3) = P(3) + P(2); P(2) = 0;  
        xk = [200.2; 216.2; 0; 0]; uk = [0.8; 3063];
        A(1,o+n) = lmbda(gamma, b0_2, alpha, n);
        A(1,o+n) = 0;
        for i = 0:n
            A(o+i,2) = lmbda(gamma, b0_1, alpha, n-i);
        end
        A = fixA(A);
        tau = eta * tau;
        tau_c = tau_c + tau;
    end
        
    % Compute health metrics
    PFD(j+1) = P(2);
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
tauC = ceil(tau0); tau_s = tau0;
for i = 2:n
    tau_s = eta * tau_s;
    tauC(i) = ceil(tauC(i-1) + tau_s);
end
    
%% State Tracking Plot
fh1 = figure(1); clf;
tiledlayout(2,1, 'Padding', 'compact', 'TileSpacing', 'compact');

% Chlorine Flow
nexttile
plot(times, xhist(3,1:end-1), 'LineWidth', 2, 'Color', [0 0.45 0.74]); hold on
yline(1.708e-3, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.85 0.33 0.1]);
xlim([0 times(end)]); ylim([0.1e-2 0.3e-2])
xticks([tauC(1) tauC(3) tauC(5) tauC(7) tauC(9) tauC(11) tauC(13) tauC(15)]); 
xticklabels({'\tau_0','\tau_2','\tau_4','\tau_6','\tau_8','\tau_{10}', ...
    '\tau_{12}','\tau_{14}'})
legend({'$q_c\;(t)$','$q_c^{ref}$'}, 'Interpreter', 'latex')
ylabel('$q_c\;(m^3h^{-1})$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

% Feed Pressure
nexttile
plot(times, xhist(4,1:end-1), 'LineWidth', 2, 'Color', [0.47 0.67 0.19]); hold on
yline(230.2, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.85 0.33 0.1]);
xlim([0 times(end)]); ylim([220 240])
xticks([tauC(1) tauC(3) tauC(5) tauC(7) tauC(9) tauC(11) tauC(13) tauC(15)]); 
xticklabels({'\tau_0','\tau_2','\tau_4','\tau_6','\tau_8','\tau_{10}', ...
    '\tau_{12}','\tau_{14}'})
legend({'$p_f\;(t)$','$p_f^{ref}$'}, 'Interpreter', 'latex')
ylabel('$p_f\;(bar)$', 'Interpreter','latex')
xlabel('$t\;(h)$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

%% Control & Risk Metrics Plot
fh2 = figure(2); clf;
tiledlayout(3,1, 'Padding', 'compact', 'TileSpacing', 'compact');

% PFD
nexttile
plot(times, PFD(1:end-1), 'LineWidth', 2, 'Color', [0.49 0.18 0.56]); hold on
yline(PFDAvg, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.85 0.33 0.1]);
xlim([0 times(end)])
xticks([tauC(1) tauC(3) tauC(5) tauC(7) tauC(9) tauC(11) tauC(13) tauC(15)]); 
xticklabels({'\tau_0','\tau_2','\tau_4','\tau_6','\tau_8','\tau_{10}', ...
    '\tau_{12}','\tau_{14}'})
legend({'$\mathrm{PFD}\;(t)$','$\mathrm{PFD}$'}, 'Interpreter', 'latex')
ylabel('$\mathrm{PFD}$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

% Control Inputs
nexttile
plot(times, nuhist(1,:), 'LineWidth', 2, 'Color', [0.93 0.69 0.13]); hold on
plot(times, nuhist(2,:), 'LineWidth', 2, 'Color', [0.3 0.75 0.93]);
xlim([0 times(end)]); ylim([0 1]);
xticks([tauC(1) tauC(3) tauC(5) tauC(7) tauC(9) tauC(11) tauC(13) tauC(15)]); 
xticklabels({'\tau_0','\tau_2','\tau_4','\tau_6','\tau_8','\tau_{10}', ...
    '\tau_{12}','\tau_{14}'})
legend({'$z^*\;(t)$','$\omega^*\;(t)$'}, 'Interpreter', 'latex')
ylabel('$u^*$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)

% Hazard Rate
nexttile
plot(times, HR, 'LineWidth', 2, 'Color', [0.64 0.08 0.18]); hold on
yline(HRAvg, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0 0.45 0.74]);
yline(uHR, 'LineWidth', 2, 'LineStyle', ':', 'Color', [0.47 0.67 0.19]);
xlim([0 times(end)]); ylim([0 2e-8])
xticks([0 tauC(1) tauC(2) tauC(3) tauC(4) tauC(5) tauC(6) tauC(7) tauC(8) ...
    tauC(9) tauC(10) tauC(11) tauC(12) tauC(13) tauC(14) tauC(15) tauC(16)]); 
xticks([tauC(1) tauC(3) tauC(5) tauC(7) tauC(9) tauC(11) tauC(13) tauC(15)]); 
xticklabels({'\tau_0','\tau_2','\tau_4','\tau_6','\tau_8','\tau_{10}', ...
    '\tau_{12}','\tau_{14}'})
legend({'$\mathrm{HR}\;(t)$','$\mathrm{HR}$','$\overline{\mathrm{HR}}$'}, ...
    'Interpreter','latex')
ylabel('$\mathrm{HR}\;(h^{-1})$', 'Interpreter','latex')
xlabel('$t\;(h)$', 'Interpreter','latex')
grid on; set(gca, 'FontSize', 14)


%% Utility Functions
function tree = DistTree(dval_list, Nr)
    Nd = numel(dval_list);
    
    % Generate all combinations for one time step
    [G{1:Nd}] = ndgrid(dval_list{:});
    one_step = reshape(cat(Nd+1, G{:}), [], Nd);
    Nv = size(one_step, 1);

    % Total scenarios
    total_scenarios = Nv^Nr;
    
    % Generate all index combinations for Nr steps
    idx_cells = cell(1, Nr);
    [idx_cells{:}] = ndgrid(1:Nv);
    idx_combo = reshape(cat(Nr+1, idx_cells{:}), [], Nr);

    % Build the tree
    tree = zeros(total_scenarios, Nd, Nr);
    for t = 1:Nr
        tree(:,:,t) = one_step(idx_combo(:,t), :);
    end
end

function lambda_val = lmbda(gamma, b0, beta, n)
    % Failure rate as function of n
    b = b0 * exp(-beta * n);
    lambda_val = gamma * (1 - b / (b + 1));
end

function A = fixA(A)
    % Fill in the diagonal elements of a transition matrix
    N = size(A,1);
    for i = 1:N
        s = 0;
        for j = 1:N
            if i ~= j
                s = s + A(i,j);
            end
        end
        A(i,i) = -s;
    end
end

