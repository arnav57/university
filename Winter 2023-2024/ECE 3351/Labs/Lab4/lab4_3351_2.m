%% QUESTION 2

% define the difference equation
b = [1, -0.8]; % numerator (output) y coeffs
a = 1; % denominator (input) x coeffs

% find step resp
t = -20:200; % time scale (x-axis)
u = ones(size(t));
s, final = filter(b, a, u); % filters data in 't' according to num 'b' and den 'a', outputs this in 'y'

% plot step resp
stem(t,s);
grid on;
title('Step Response');
ylabel('s(n)');
xlabel('n');
