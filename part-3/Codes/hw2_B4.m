clear all;
load box1.mat                           % Loads the Distance matrix
% In case you want the Distance Matrix to be calculted, Comment out the 
% previous line-"load box1.mat" and uncomment the following.
% [D] = D_gen('box1');
%[data, names] = loadImageData('box_random', 'jpg');
%D = L2_distance(data,data,1);
options.dims = 1:10;
[Y, R, E] = Isomap(D, 'k', 5, options);