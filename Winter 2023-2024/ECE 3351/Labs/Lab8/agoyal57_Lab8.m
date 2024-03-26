clear;
%% Obtain conj pole locations
r = 0.65; % radial distance
dc_gain = 0.195;

pang = pi/10; % pole angle
xp = r * cos(pang);
yp = r * sin(pang);


%% Low Pass Filter
num = dc_gain;
% expression to guarantee conj poles.
% derived very painfully :(
den = [1, -2*xp, xp^2 + yp^2]; 

disp(num);
disp(den);

% uncomment to see everything
fvtool(num, den) 

%% Implement the transfer function
figure; 
hold on;
H = @(w) 0.1950 / (1 - 1.2364 * exp(-1i*w) + 0.4255 * exp(-2*1i*w));
% initial signal
t = 0 : 0.1 : 25;
x = sin(pi/10 * t) + sin(pi*t);
% filtered signal
y = abs(H(pi/10))*sin(pi/10 * t + angle(H(pi/10))) + abs(H(pi))*sin(pi*t + angle(H(pi)));
plot(t, x, 'blue');
plot(t, y, 'green');
legend('input', 'output');
title('Input signal vs Output signal');