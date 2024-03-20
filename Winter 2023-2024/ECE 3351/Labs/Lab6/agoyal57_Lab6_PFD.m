%% ensure clear data
clear;

%% define both system with their coefficients
num1 = [3.5, -4.75, 1.58];
den1 = [1, -1.9, 1.16, -0.224];
num2 = [2.5, -0.6, -0.8];
den2 = [1, -0.3, -0.2, 0.35];

%% obtain partial fractions &  write output to file 'output.txt'
file = fopen('output.txt', 'w');
printSystem(num1, den1, 1, file);
printSystem(num2, den2, 2, file);

%% define write to output file function
function printArr(array, label, file)
    fprintf(file, '%s:\n', label);
    for i = 1:length(array)
        fprintf(file, '%.3f + %.3fj\n', real(array(i)), imag(array(i)));
    end
    fprintf(file, '\n');
end

%% helper function for above
function printSystem(num, den, label, file)
    fprintf(file, 'SYSTEM %d\n', label);
    [r, p, k] = residuez(num, den);
    printArr(r, 'Residues', file);
    printArr(p, 'Poles', file);
    printArr(k, 'Direct Terms', file);
    fprintf(file, '---------------\n');

end