hold on;

n2 = [0:0.01:0.6];
y2 = 1*sin(10*pi*n2);
stem(n2,y2);

n3 = [0:(1/60):0.6];
y3 = 1*sin(10*pi*n3);
stem(n3,y3);

n4 = [0:(1/30):0.6];
y4 = 1*sin(10*pi*n4);
stem(n4,y4);

n5 = [0:(1/12):0.6];
y5 = 1*sin(10*pi*n5);
stem(n5,y5);

n6 = [0:(1/6):0.6];
y6 = 1*sin(10*pi*n6);
stem(n6,y6)

% U can reconstruct every signal except the one sampled at 6Hz