images = loadMNISTImages('/Users/Home/research/datasets/MNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('/Users/Home/research/datasets/MNIST/train-labels-idx1-ubyte');
images=images(:,1:3000);
labels=labels(1:3000,:);
imag=[];lab=[];
for i=1:3000
if(labels(i,:)==1 || labels(i,:)==7)
imag=[imag,images(:,i)];
lab=[lab,labels(i,:)];
end
end
D=L2_distance(imag,imag);
options.dims=1:10;
[Y,R,E]=isomap2(D,lab,'k',7,options);

imag=[];lab=[];
for i=1:3000
if(labels(i,:)==4 || labels(i,:)==9)
imag=[imag,images(:,i)];
lab=[lab,labels(i,:)];
end
end
D=L2_distance(imag,imag);
options.dims=1:10;
[Y,R,E]=isomap2(D,lab,'k',7,options);

D=L2_distance(images,images);
options.dims=1:10;
[Y,R,E]=isomap2(D,labels,'k',7,options);