"""
@Author : Mohit Jain
@Email  : develop13mohit@gmail.com
"""

import os
import sys
import random
import numpy as np
import Image
from sklearn import cross_validation, metrics
from sklearn.neighbors import KNeighborsClassifier

NAMESPACE = "/Users/Home/Sem7/Machine\ Learning/Assignment\ 2/"
mnist_path = '/Users/Home/research/datasets/MNIST'

train_file = os.path.join(mnist_path,'train/new_train.txt')
test_file = os.path.join(mnist_path,'test/test.txt')
label_file = os.path.join(mnist_path,'train/labels.txt')

train_images = []
train_labels = []
test_images = []
test_labels = []
labels = []

print "[INFO] Loading training and testing images and labels"

try:
    data = np.load(os.path.join(mnist_path,'train','training_data.npz'))
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
            i = Image.open(os.path.join(mnist_path,'train',img_path))
            train_images.append(np.array(i.getdata(),np.uint8).reshape(i.size[1], i.size[0]))
            train_labels.append(label)
            sys.stdout.flush()
    print ""
    print "[INFO] Iterative loading of training data complete. Saving data as numpy array."

    np_train_images = np.array(train_images)
    np_train_labels = np.array(train_labels)
    np.savez(os.path.join(mnist_path,'train','training_data.npz'), train_images=np_train_images, train_labels=np_train_labels)
    print "[SUCCESS] Training data successfull stored as numpy."
    print "#Train-Images : %d %s; #Train-Labels : %d; #Labels : %d" % (len(train_images), str(train_images[0].shape), len(train_labels), len(labels))

print "[INFO] Training the NearestNeighbour model."
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
clf.fit(train_images, train_labels)	

print "[SUCCESS] Training Complete"
expected = train_labels
predicted = clf.predict(train_images)

print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted))
print "Confusion matrix:\n", metrics.confusion_matrix(expected, predicted)

# for i in random.sample(xrange(len(train_images)),10):
# 	print clf.predict([train_images[i]]), train_labels[i]

# print "[OUTPUT] Scores on cross-validation."
# """ Train and validate the model with 7-fold cross validation """
# scores = cross_validation.cross_val_score(clf, train_images, train_labels, cv=7)
# print scores