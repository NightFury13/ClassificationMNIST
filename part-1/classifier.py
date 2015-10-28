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
parser.add_argument('-clf_type',dest='clf_type',help='Type of classifier to use [SVM, NN]')
parser.add_argument('-kernel_type',dest='kernel_type',help='Type of kernel to use [linear, rbf, Chi-square, poly, sigmoid]')
parser.add_argument('-report',dest='report',type=bool,help='Display report after training?')
parser.add_argument('-conf_mat',dest='conf_mat',type=bool,help='Display confusion matrix after training?')
parser.add_argument('-C',dest='C',type=float,help='Value of classifier constant C')
parser.add_argument('-gamma',dest='gamma',type=float,help='Value of gamma for classifier training.')
parser.add_argument('-neigh',dest='neigh',type=int,help='#Neighbours to consider for NN model.')
args = parser.parse_args()

if args.data_path == None or args.clf_type == None or args.kernel_type == None:
    err_str =  "[ERROR] Incorrect usage! Run script as follows :\n"
    err_str += "$> python classifier.py -data_path /Users/Home/research/datasets/MNIST -clf_type [SVM,NN] -kernel_type [linear, rbf, sigmoid, Chi-square, poly] -report True/False -conf_mat True/False -C 2.8 -gamma 0.0073 -neigh 5\n"
    print err_str
    raise "Invalid script Initialization"

clf_type = args.clf_type
kernel_type = args.kernel_type
show_report = True if args.report else False
show_mat = True if args.conf_mat else False
C = 2.8 if not args.C else float(args.C)
gamma = 0.0073 if not args.gamma else float(args.gamma)
neigh = 5 if not args.neigh else int(args.neigh)
data_path = '/Users/Home/research/datasets/MNIST' if not args.data_path else str(args.data_path)
#-----------------------------------------------------------------#

#------------------ Global Variables Init ------------------------#
valid_clf = ['SVM', 'NN']
valid_kernel = ['linear','rbf','Chi-square','poly','sigmoid']

if clf_type not in valid_clf:
	print "[ERROR] Invalid classifier type. Choose from %s\n" % str(valid_clf)
	raise "Invalid Classifier Type"
if kernel_type not in valid_kernel:
	print "[ERROR] Invalid kernel type. Choose from %s" % str(valid_clf)
	raise "Invalid Kernel Type"

train_file = os.path.join(data_path,'train/new_train.txt')
test_file = os.path.join(data_path,'test/new_test.txt')
label_file = os.path.join(data_path,'train/labels.txt')

train_images = []
train_labels = []
test_images = []
test_labels = []
labels = []
#-----------------------------------------------------------------#

#--------------- Training Dataset Loading ------------------------#
print "[INFO] Loading training images and labels"

try:
    data = np.load(os.path.join(data_path,'train','training_data.npz'))
    train_images = list(data['train_images'].reshape(data['train_images'].shape[0], -1))
    train_labels = list(data['train_labels'])
    with open(label_file) as f:
    	for line in f.readlines():
    		labels.append(int(line.strip()))
    print "[SUCCESS] Training data successfully loaded from numpy."
    print "     #Train-Images : %d %s; #Train-Labels : %d; #Labels : %d" % (len(train_images), str(train_images[0].shape), len(train_labels), len(labels))
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
            train_images.append(np.array(i.getdata(),np.uint8).reshape(i.size[1], i.size[0]))
            train_labels.append(label)
            sys.stdout.flush()
    print ""
    print "[INFO] Iterative loading of training data complete. Saving data as numpy array."

    np_train_images = np.array(train_images)
    np_train_labels = np.array(train_labels)
    np.savez(os.path.join(data_path,'train','training_data.npz'), train_images=np_train_images, train_labels=np_train_labels)
    print "[SUCCESS] Training data successfull stored as numpy."
    print "#Train-Images : %d %s; #Train-Labels : %d; #Labels : %d" % (len(train_images), str(train_images[0].shape), len(train_labels), len(labels))
#--------------------------------------------------------------------#

#----------------- Testing Data Loading -----------------------------#
print "[INFO] Loading Testing images and labels"

try:
    data = np.load(os.path.join(data_path,'test','testing_data.npz'))
    test_images = list(data['test_images'].reshape(data['test_images'].shape[0], -1))
    test_labels = list(data['test_labels'])
    print "[SUCCESS] Training data successfully loaded from numpy."
    print "     #Test-Images : %d %s; #Test-Labels : %d" % (len(test_images), str(test_images[0].shape), len(test_labels))
except:
    print "[FAIL] Loading testing_data from numpy array failed. Manually loading data."
    idx = 0.0
    len_tot = 60000
    with open(test_file,'r') as test_data:
        for line in test_data.readlines():
            idx += 1
            sys.stdout.write("\r[LOAD] [%0.3f%s] Loading %s" %((idx*100)/len_tot,'%',line.strip()))
            img_path = line.split()[0].strip()
            label = int(line.split()[1].strip())
            i = Image.open(os.path.join(data_path,'test',img_path))
            test_images.append(np.array(i.getdata(),np.uint8).reshape(i.size[1], i.size[0]))
            test_labels.append(label)
            sys.stdout.flush()
    print ""
    print "[INFO] Iterative loading of test data complete. Saving data as numpy array."

    np_test_images = np.array(test_images)
    np_test_labels = np.array(test_labels)
    np.savez(os.path.join(data_path,'test','testing_data.npz'), test_images=np_test_images, test_labels=np_test_labels)
    print "[SUCCESS] Testing data successfull stored as numpy."
    print "#Test-Images : %d %s; #Test-Labels : %d" % (len(test_images), str(test_images[0].shape), len(test_labels))
#--------------------------------------------------------------------#

#---------------------- Perform Classification ----------------------#
print "[INFO] Training the model on loaded data."
if clf_type=='SVM':
	clf = SVC(kernel=kernel_type, C=C, gamma=gamma)
elif clf_type=='NN':
	clf = KNeighborsClassifier(n_neighbors=neigh, weights='distance')

clf.fit(train_images, train_labels)	

print "[SUCCESS] Training Complete"
expected = test_labels
predicted = clf.predict(test_images)
#--------------------------------------------------------------------#

#--------------------- View Outputs ---------------------------------#
if show_report:
	print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted))
if show_mat:	
	print "Confusion matrix:\n", metrics.confusion_matrix(expected, predicted)
#--------------------------------------------------------------------#