#!/bin/bash
#SBATCH --mem=10000
#SBATCH -n 1
#SBATCH -t 320:00:00
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out

python classifier.py -data_path /Pulsar2/mohit.jain/datasets/MNIST -clf_type SVM -kernel_type sigmoid -report True -conf_mat True -out_file SVM_sigmoid_out.txt

