#!/bin/bash
#SBATCH --mem=10000
#SBATCH -n 1
#SBATCH -t 320:00:00
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out

python modified_training.py -data_path /Pulsar2/mohit.jain/datasets/MNIST -percent 10 -C 2.8 -gamma 0.0073 -delta 10.0

