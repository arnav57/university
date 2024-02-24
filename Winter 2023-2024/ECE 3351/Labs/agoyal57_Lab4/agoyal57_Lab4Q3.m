%% QUESTION 3

% define the difference equation
b = 1; % numerator (output) x coeffs
a = [1, -1, 0.8]; % denominator (input) y coeffs

% finding impulse resp
t = -20:200;
[h, n] = impz(b, a, length(t));

% check stability criterion, does the summation converge?
val = sum(abs(h));
stable = val < inf; % this is a boolean 1=True, 0=False
disp(['System Stable? : ' num2str(stable)]);
disp(['Converges to : ' num2str(val) ' on -20:200']);