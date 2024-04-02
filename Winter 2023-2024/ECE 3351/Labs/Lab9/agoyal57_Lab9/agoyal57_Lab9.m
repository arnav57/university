clear;
%% Convert to radial frequencies
fs = 400;
w = 2*pi*120/fs; % this is the rad freq we have to null

%% Design notch filter
r = 0.9; % mag of pole
k = 0.9; % gain of num
num = k*[1, -2*cos(w), 1];
den = [1, -2*r*cos(w), r^2];

w_20 = 2*pi*20/fs;
w_180 = 2*pi*180/fs;

fvtool(num, den)

%% Demonstrate Conformity to Specifications
h = freqz(num, den, [w_20, w_180]);
str = sprintf("Gain of 20 Hz: %f\nGain of 180 Hz %f", abs(h(1)), abs(h(2)));
fprintf("\nConformity to Specifications:\n");
disp(str);

%% Sample Input Signal
t = 0 : 0.1 : 35;
w0 = 2*pi*20/fs;
w1 = 2*pi*120/fs;
w2 = 2*pi*180/fs;

h = freqz(num, den, [w0, w1, w2]);

x = sin(w0*t) + sin(w1*t) + sin(w2*t);
y = abs(h(1))*sin(w0*t + angle(h(1))) + abs(h(2))*sin(w1*t + angle(h(2))) + abs(h(3))*sin(w2*t + angle(h(3)));

figure;
hold on;
plot(t, x, 'blue');
plot(t, y, 'red');
legend('input', 'output');
title('Input vs Output Signal')