%% QUESTION 1
clear;

% define the difference equation
b = 1; % numerator (output) x coeffs
a = [1, -1, 0.8]; % denominator (input) y coeffs

% finding impulse resp
t = -20:200;
[h, n] = impz(b, a, length(t));

% plotting impulse resp
stem(n, h);
grid on;
title('Impulse Response');
xlabel('n');
ylabel('h(n)');