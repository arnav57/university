input = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
% any impulses

method1 = zeros(size(input));
method2 = zeros(size(input));
method3 = zeros(size(input));

% method 1
for i = 4:numel(input)
    method1(i) = mean(input(i-3:i));
end

% method 2
for i = 2:numel(input)
    method2(i) = (0.3 * input(i)) + (0.7*method2(i-1));
end

% method 3
for i = 5:numel(input)
    method3(i) = (0.25) * (input(i) - input(i-4)) + method3(i-1);
end

% plotting
figure;
hold on;
plot(method1, 'r');
plot(method2, 'g');
plot(method3, 'b');
plot(input, 'black');
title('Impulse Response of All Methods')
xlabel('time')
ylabel('v avg(n)')