clear;
% define the systems
num1 = [3.5, -0.75, -0.25];
den1 = [1, -0.5, -0.25, 0.125];

num2 = [7 -12.2 -6.9 -1.5];
den2 = [1 -2.8 3.1 -1.7 0.4];

num3 = [5 -1.5 0.8125];
den3 = [1 -1 0.8125];

num4 = [5 -20 31.75 -18.5];
den4 = [1 -6 13.25 -13 5];

% create output file
file = fopen('output.txt', 'w');

% get outputs (see functions)
printSystem(num1, den1, 1, file);
printSystem(num2, den2, 2, file);
printSystem(num3, den3, 3, file);
printSystem(num4, den4, 4, file);


% funcs
function printArr(array, label, file)
    fprintf(file, '%s:\n', label);
    for i = 1:length(array)
        fprintf(file, '%.3f + %.3fj\n', real(array(i)), imag(array(i)));
    end
    fprintf(file, '\n');
end

function printSystem(num, den, label, file)
    fprintf(file, 'SYSTEM %d\n', label);
    [r, p, k] = residuez(num, den);
    printArr(r, 'Residues', file);
    printArr(p, 'Poles', file);
    printArr(k, 'Direct Terms', file);
    fprintf(file, '---------------\n');

end