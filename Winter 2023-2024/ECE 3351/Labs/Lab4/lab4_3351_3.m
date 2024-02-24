%% QUESTION 3

% define the difference equation
b = [1, -0.8]; % numerator (output) y coeffs
a = 1; % denominator (input) x coeffs

% finding impulse resp
t = -20:200;
[h, n] = impz(b, a, length(t));

% check stability criterion, does the summation converge?
stable = sum(abs(h)) < inf; % this is a boolean 1=True, 0=False
disp(['System Stability: ' num2str(stable)]); % 1 means stable, 0 means unstable