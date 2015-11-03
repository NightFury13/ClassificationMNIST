"""
@Author : Mohit Jain (NightFury13)
@Email  : develop13mohit@gmail.com
"""

import os
import sys
import random
import argparse
import numpy as np
import Image
from sklearn import cross_validation, metrics
from sklearn.svm import SVC

#-------------------- Handle Input --------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('-data_path',dest='data_path',help='Path to the dataset folder')
parser.add_argument('-percent',dest='percent',type=float,help='Percantage of data to select for initial training')
parser.add_argument('-C',dest='C',type=float,help='Value of classifier constant C')
parser.add_argument('-gamma',dest='gamma',type=float,help='Value of gamma for classifier training.')
parser.add_argument('-delta',dest='delta',type=float,help='Constant (min=0,max=1) to consider for stopping training iterations.')
args = parser.parse_args()

if args.data_path == None:
    err_str =  "[ERROR] Incorrect usage! Run script as follows :\n"
    err_str += "$> python modified_training.py -data_path /Users/Home/research/datasets/MNIST -percent 10 -C 2.8 -gamma 0.0073 -delta 0.01\n"
    print err_str
    raise "Invalid script Initialization"

percent = float(args.percent) if args.percent else 10.0 
C = 2.8 if not args.C else float(args.C)
gamma = 0.0073 if not args.gamma else float(args.gamma)
delta = 0.01 if not args.delta else float(args.delta)
data_path = '/Users/Home/research/datasets/MNIST' if not args.data_path else str(args.data_path)
#-----------------------------------------------------------------#

#------------------ Global Variables Init ------------------------#
train_file = os.path.join(data_path,'train/new_train.txt')
train_images = []
train_labels = []
test_images = []
test_labels = []
loaded_labels = []
loaded_images = []
#-----------------------------------------------------------------#

#--------------- Training Dataset Loading ------------------------#
print "[INFO] Loading training images and labels"

try:
    data = np.load(os.path.join(data_path,'train','training_data.npz'))
    loaded_images = list(data['train_images'].reshape(data['train_images'].shape[0], -1))
    loaded_labels = list(data['train_labels'])
    
    train_idx = int(len(loaded_images)*(percent/100))
    train_images = loaded_images[:train_idx]
    train_labels = loaded_labels[:train_idx]
    test_images = loaded_images[train_idx+1:]
    test_labels = loaded_labels[train_idx+1:]
    print "[SUCCESS] Training data successfully loaded from numpy."
    print "     #Train-Images : %d %s; #Train-Labels : %d" % (len(train_images), str(train_images[0].shape), len(train_labels))
    print "     #Test-Images : %d %s; #Test-Labels : %d" % (len(test_images), str(test_images[0].shape), len(test_labels))
except:
    print "[FAIL] Loading training_data from numpy array failed. Manually loading data."

    idx = 0.0
    len_tot = 60000
    with open(train_file,'r') as train_data:
        for line in train_data.readlines():
            idx += 1
            sys.stdout.write("\r[LOAD] [%0.3f%s] Loading %s" %((idx*100)/len_tot,'%',line.strip()))
            img_path = line.split()[0].strip()
            label = int(line.split()[1].strip())
            i = Image.open(os.path.join(data_path,'train',img_path))
            loaded_images.append(np.array(i.getdata(),np.uint8).reshape(i.size[1], i.size[0]))
            loaded_labels.append(label)
            sys.stdout.flush()
    print ""
    print "[INFO] Iterative loading of training data complete. Saving data as numpy array."
    np_loaded_images = np.array(loaded_images)
    np_loaded_labels = np.array(loaded_labels)
    np.savez(os.path.join(data_path,'train','training_data.npz'), train_images=np_train_images, train_labels=np_train_labels)
    print "[INFO] Training data successfull stored as numpy."

    train_idx = int(len(loaded_images)*(percent/100))
    train_images = loaded_images[:train_idx]
    train_labels = loaded_labels[:train_idx]
    test_images = loaded_images[train_idx+1:]
    test_labels = loaded_labels[train_idx+1:]
    print "[SUCCESS] Training data successfully loaded to memory."
    print "     #Train-Images : %d %s; #Train-Labels : %d" % (len(train_images), str(train_images[0].shape), len(train_labels))
    print "     #Test-Images : %d %s; #Test-Labels : %d" % (len(test_images), str(test_images[0].shape), len(test_labels))
#--------------------------------------------------------------------#

#---------------------- Perform Classification ----------------------#
print "[INFO] Training the model on loaded data %s%s of training data." % (str(percent),'%')

iteration = 0
percent_incorrect = 1 # Init value
clf = SVC(C=C, gamma=gamma)

while percent_incorrect > delta:
    iteration += 1
    print "[ Iteration %s ] :: #Train-Images : %s; #Test-Images : %s" % (iteration, str(len(train_images)), str(len(test_images)))
    print "    [INFO] Training classifier on training images."
    clf.fit(train_images, train_labels)
    
    expected = test_labels
    predicted = clf.predict(test_images)
    incorrect_indices = []
    for i in range(len(test_images)):
        if expected[i]!=predicted[i]:
            incorrect_indices.append(i)
    percent_incorrect = float(len(incorrect_indices))/len(test_images)
    print "    [INFO] %s%s(%s%s) images classified incorrectly." % (str(len(incorrect_indices)), '%', percent_incorrect, '%')

    for i in range(len(incorrected_indices)):
        train_images.append(test_images[i])
        train_labels.append(test_labels[i])
        test_images.pop(i)
        test_labels.pop(i)
#--------------------------------------------------------------------#

#--------------------- View Outputs ---------------------------------#
print ""
print "[COMPLETE] Training Process Complete."
print "[ Final ] %s%s(%s%s) images classified incorrectly." % (str(len(incorrect_indices)), '%', percent_incorrect, '%')
#--------------------------------------------------------------------#
