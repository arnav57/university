%% QUESTION 1

% define the difference equation
b = [1, -0.8]; % numerator (output) y coeffs
a = 1; % denominator (input) x coeffs

% finding impulse resp, should look like a decaying exp
t = -20:200;
[h, n] = impz(b, a, t);

% plotting impulse resp
stem(t, h);
grid on;
title('Impulse Response');
xlabel('n');
ylabel('h(n)');