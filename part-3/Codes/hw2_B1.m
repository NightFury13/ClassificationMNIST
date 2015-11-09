clear all;
load nao_distance.mat                           % Loads the Distance matrix
% In case you want the Distance Matrix to be calculted, Comment out the 
% previous line-"load nao_distance.mat" and uncomment the following.
% [D] = D_gen('nao_images')
options.dims = 1:10;
[Y, R, E] = Isomap(D, 'k', 12, options);