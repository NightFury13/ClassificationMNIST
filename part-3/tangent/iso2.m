images = loadMNISTImages('/Users/Home/research/datasets/MNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('/Users/Home/research/datasets/MNIST/train-labels-idx1-ubyte');
images=images(:,1:3000);
labels=labels(1:3000,:);
imag=[];lab=[];

D=tangent_d(images,images);
options.dims=2;
[Y,R,E]=isomap2(D,labels,'k',7,options);