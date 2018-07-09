%correlation between two matrics
image = imread("redEyeTreeFrog.jpg");
image = image(:,:,1);

%calculating the sigmoid activation
exponential = -1*(image);
den = 1 + exp(exponential);
sigmoid = 1/den;

%tanh activation function
exponential = -2*(image);
den = 1+exp(exponential);
tanh = 2/den;

tanh
sigmoid
