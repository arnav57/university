%% QUESTION 2
clear;

% define the difference equation
b = 1; % numerator (output) x coeffs
a = [1, -1, 0.8]; % denominator (input) y coeffs

% find step resp
t = -20:200; % time scale (x-axis)
u = ones(size(t));
[s, final] = filter(b, a, u); % filters data in 't' according to num 'b' and den 'a', outputs this in 'y'

% plot step resp
stem(t,s);
grid on;
title('Step Response');
ylabel('s(n)');
xlabel('n');
