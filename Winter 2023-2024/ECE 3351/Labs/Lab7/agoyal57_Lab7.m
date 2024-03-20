%% Frequency Response 
clear;
% define the LTI system
num = [1, 1, 0.5];
den = [1, -1.7, 1.2, -0.35];

% define the transfer function H(w): R -> C
H = @(w) (1 + exp(-w*1i) + 0.5*exp(-2*w*1i)) / (1 - 1.7*exp(-w*1i) +1.2*exp(-2*w*1i) -0.35*exp(-3*w*1i));

% obtain magnitudes and phases w.r.t frequency
w = linspace(0, pi, 100);
w_labels = w/pi;
mags = zeros(size(w));
angs = zeros(size(w));
for i = 1:length(w)
    h_val = H(w(i));
    mags(i) = abs(h_val);
    angs(i) = rad2deg(angle(h_val));
end

% plot mags and angs, cant use freqz as it returns dB gain not u/u gain
figure;

subplot(2,1,1);
plot(w/pi, mags, 'magenta');
grid on;
title('Magnitude Response');
xlabel('Normalized Frequency [\pi rad/sec]');
ylabel('Magnitude [unit/unit]');
subplot(2,1,2);
plot(w/pi, angs, 'magenta');
grid on;
title('Phase Response');
xlabel('Normalized Frequency [\pi rad/sec]');
ylabel('Phase Shift [deg]');

%% Find Responses to x1 and x2
n = 0:99; % timestep

% get phasors for frquencies of sinusoids
H1 = H(pi/6);
H2 = H(pi/3);
H3 = H(5*pi/6);
H4 = H(2*pi/3);

x1 = abs(H1) * cos(n*pi/6 + angle(H1)) + abs(H2) * cos(n*pi/3 + angle(H2));
x2 = abs(H3) * cos(5*n*pi/6 + angle(H3)) + abs(H4) * cos(2*n*pi/3 + angle(H4));

% plot the responses
figure;
subplot(2,1,1);
stem(x1, 'magenta');
grid on;
title('Response to x_1(n)');
xlabel('Timestep, n');
ylabel('y(n)');
subplot(2,1,2);
stem(x2, 'magenta');
grid on;
title('Response to x_2(n)');
xlabel('Timestep, n ');
ylabel('y(n)');


