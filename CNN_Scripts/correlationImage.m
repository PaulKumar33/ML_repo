%correlation between two matrics
imText = 'redEyeTreeFrog.jpg';
image = imread(imText);
image = image(:,:,2);
image = double(image);
%calculating the sigmoid activation
exponential1 = -(image);
exponential1 = normalize(exponential1);
%exponential = int(exponential);
den1 = 1+exp((exponential1));
sigmoid = 1./den1;

%tanh activation function
exponential2 = -1*(image);
exponential2 = normalize(exponential2);
exponential2 = 2*exponential2;
den2 = 1+exp(double(exponential2));
tanh = 2./den2;

corrAB = corr2(uint8(tanh), uint8(sigmoid));

