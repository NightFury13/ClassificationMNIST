usage:$> python classifier.py [-h] [-data_path DATA_PATH] [-clf_type CLF_TYPE]
                     [-kernel_type KERNEL_TYPE] [-report REPORT]
                     [-conf_mat CONF_MAT] [-C C] [-gamma GAMMA] [-neigh NEIGH]

optional arguments:
  -h, --help            show this help message and exit
  -data_path DATA_PATH  Path to the dataset folder
  -clf_type CLF_TYPE    Type of classifier to use [SVM, NN]
  -kernel_type KERNEL_TYPE
                        Type of kernel to use [linear, rbf, Chi-square, poly,
                        sigmoid]
  -report REPORT        Display report after training?
  -conf_mat CONF_MAT    Display confusion matrix after training?
  -C C                  Value of classifier constant C
  -gamma GAMMA          Value of gamma for classifier training.
  -neigh NEIGH          #Neighbours to consider for NN model.
