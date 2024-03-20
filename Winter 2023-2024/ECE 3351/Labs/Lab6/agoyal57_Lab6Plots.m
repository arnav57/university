%% ensure clear data
clear;

%% define both system with their coefficients
num1 = [3.5, -4.75, 1.58];
den1 = [1, -1.9, 1.16, -0.224];
num2 = [2.5, -0.6, -0.8];
den2 = [1, -0.3, -0.2, 0.35];

%% get imp/step responses
imp1 = filter(num1, den1, [1, zeros(1, 50)]);
step1 = filter(num1, den1, ones(1, length(imp1)));
imp2 = filter(num2, den2, [1, zeros(1, 50)]);
step2 = filter(num2, den2, ones(1, length(imp2)));

%% Plot responses of system 1
figure;
subplot(2,1,1);
stem(0:50, imp1);
title('Impulse Response - System 1');
xlabel('n');
ylabel('Amplitude');

subplot(2,1,2);
stem(0:50, step1);
title('Step Response - System 1');
xlabel('n');
ylabel('Amplitude');

%% Plot responses of system 2
figure;
subplot(2,1,1);
stem(0:50, imp2);
title('Impulse Response - System 2');
xlabel('n');
ylabel('Amplitude');

subplot(2,1,2);
stem(0:50, step2);
title('Step Response - System 2');
xlabel('n');
ylabel('Amplitude');

%% Plot Zero-Pole of System 1
figure;
zplane(num1, den1);
legend('Zero', 'Pole')
title('Zero Pole Plot - System 1');

%% Plot Zero-Pole of System 2
figure;
zplane(num2, den2);
legend('Zero', 'Pole')
title('Zero Pole Plot - System 2');
