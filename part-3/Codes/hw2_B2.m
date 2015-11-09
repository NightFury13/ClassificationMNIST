clear all;
load box_random.mat                           % Loads the Distance matrix
% In case you want the Distance Matrix to be calculted, Comment out the 
% previous line-"load box_random.mat" and uncomment the following.
% [D] = D_gen('box_random');
%[data, names] = loadImageData('box_random', 'jpg');
%D = L2_distance(data,data,1);
options.dims = 1:20;
[Y, R, E] = Isomap(D, 'k', 12, options);