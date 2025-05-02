clear; close all; clc;

%% Parameters
gamma = 4e-5; b0 = 5; alpha = 0.5; beta = 0.3; lamf = 4e-7;        
T = 26280; maxM = 200; dt = 1;        
P_init = [0 0 0 1];   
hbar   = waitbar(0, "Simulating Test Scheduling Optimization...");

%% Preallocate
PFDAvg = zeros(1, maxM + 1);
PFDHist = zeros(maxM + 1, round(T / dt) + 2);

%% Optimization Loop
for M = 0:maxM
    if M == 0
        tau = T + 1;
    else
        tau = round(T / M);
    end
    
    t = 0;
    n = 0; m = 0;
    P = P_init;
    PFD = P(1);
    PFDCum = PFD;

    for i = 1:(T/dt)+1
        % Proof test interval (except at t = 0)
        if mod(t, tau) == 0 && t ~= 0
            m = m + 1;
        end

        % Degradation model
        b = b0 * exp(-alpha * n - beta * m);
        F = 1 - exp(-b);
        lam = gamma * (1 - F);

        % Transition matrix A (Markov model)
        A = [0         0            0           0;
             lamf+lam -(lamf+lam)   0           0;
             lamf      lam         -(lamf+lam)  0;
             lamf      0            lam        -(lamf+lam)];

        % Discretized system update
        P = P * (eye(4) + A * dt);

        % Proof test reset
        if mod(t, tau) == 0
            P = [0 P(2) P(3) P(4) + P(1)];
        end

        % Track PFD
        PFD(i + 1) = P(1);
        PFDCum = PFDCum + PFD(i + 1);
        t = t + dt;
    end

    % Store results
    PFDHist(M + 1, :) = PFD;
    PFDAvg(M + 1) = PFDCum / (T / dt + 2);

    % Update waitbar
    waitbar(M / maxM, hbar);
end

close(hbar);

%% Get Optimal Result
[optimalPFD, optimalIdx] = min(PFDAvg);
M_vals = 0:maxM;

%% Plot Results
figure('Position', [315, 520, 700, 325]);
plot(M_vals, PFDAvg, '-p', 'LineWidth', 2, ...
    'MarkerIndices', optimalIdx, 'MarkerSize', 20, ...
    'MarkerFaceColor', [1 0.5 0]); hold on;

% Add SIL targets
% yline(0.01, 'r--', 'LineWidth', 2); 
yline(0.001, 'r--', 'LineWidth', 2);

% Mark specific test count
highlightIdx = 4;
plot(M_vals(highlightIdx), PFDAvg(highlightIdx), 'o', ...
    'LineWidth', 2, 'MarkerSize', 15, ...
    'MarkerFaceColor', "#77AC30", 'Color', "#0072BD");

% Labels and annotations
xlabel('$M$', 'Interpreter', 'latex');
ylabel('$PFD$', 'Interpreter', 'latex');
grid on;
set(gca, 'FontSize', 14);

% Annotations
text(145, 0.0014, 'Lower Bound of SIL 2', 'Color', 'r', 'FontSize', 14);
text(7.42, 0.002, 'Preferred', 'FontSize', 14);
text(177, 0.0005, 'Optimal', 'FontSize', 14);
