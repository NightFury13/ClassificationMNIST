Explanation for Codes:



For Question A: The training data from MNIST database must be extracted into a folder named 'data' in the current work directory. On our system, these files 
get extracted with a slightly different name 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte'. We have made the required changes in the LoadDigits.m
to account for this.

For generating the clusters, we have extracted the points with particular required labels and then plotted them with different colors as has been explained 
with the iamge.

For generating the plots for Question A.1, run hw2_A1.m
For generating the plots for Question A.2, run hw2_A2.m




For Question B:

For questions B1,B2 and B4, We have taken a different approach then suggested. LodeImageData.m and then L2_distance.m essentially resize
the image down to a 100x100 image and then compute the Distance Matrix D. Instead, we have written a function D_gen.m, which generates the D matrix taking
from the images without resizing them. This function takes 1 image at a time and computes the row in Distance matrix D corresponding to that image, i.e.
it calculates the distance of 1 sample from all other samples in one iteration of the outer loop. this procedure has the advantage of giving more accuracy because
it is considering all pixels, but it takes much more time (time complexity increased to save space, otherwise it runs out of memory).
As this procedure takes a long time (around half an hour), we decided to do it once for all the 3 parts B1,B2 and B4 and store the corresponding Distance
matrices in
nao_data.mat
box_random.mat
box1.mat
respectively.
In the scripts for these parts, these files are loaded, and used directly.

To run the function D_gen.m the command is function [D] = D_gen(folder), where folder gives the address of the image directory. For example if the images are
stored in a folder 'box_random' in the current work directory, then the command would be

D = D_gen('box_random')

For generating the plots for Question B.1, run hw2_B1.m
For generating the plots for Question B.2, run hw2_B2.m
For generating the plots for Question B.4, run hw2_B4.m