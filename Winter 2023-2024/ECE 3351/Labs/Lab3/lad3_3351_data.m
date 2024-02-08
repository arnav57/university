load('GSPTSX.mat'); % load the data

method1 = zeros(size(SPTSX));
method2 = zeros(size(SPTSX));
method3 = zeros(size(SPTSX));

% method 1
for i = 4:numel(SPTSX)
    method1(i) = mean(SPTSX(i-3:i));
end

% method 2
for i = 2:numel(SPTSX)
    method2(i) = (0.3 * SPTSX(i)) + (0.7*method2(i-1));
end

% method 3
for i = 5:numel(SPTSX)
    method3(i) = (0.25) * (SPTSX(i) - SPTSX(i-4)) + method3(i-1);
end

for i =1:numel(method3)
    method3(i) = method3(i) + 12650;
end

figure;
hold on;
% plot(SPTSX, 'black');
% plot(method1, 'r')
plot(method2, 'g');
% plot(method3, 'b');
title('Results of All Methods on SPTSX data');
xlabel('Day');
ylabel('Index');