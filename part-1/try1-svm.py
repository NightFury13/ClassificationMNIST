"""
@Author : Mohit Jain
@Email  : develop13mohit@gmail.com
"""

import os
import sys
import numpy as np
import Image
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.datasets import load_digits

"""
NAMESPACE = "/Users/Home/Sem7/Machine\ Learning/Assignment\ 2/"
mnist_path = '/Users/Home/research/datasets/MNIST'

train_file = os.path.join(mnist_path,'train/new_train.txt')
test_file = os.path.join(mnist_path,'test/test.txt')

train_images = []
train_labels = []
test_images = []
test_labels = []

print "[INFO] Loading training and testing images and labels"

try:
    data = np.load(os.path.join(mnist_path,'train','training_data.npz'))
    train_images = list(data['train_images'])
    train_labels = list(data['train_labels'])
    print "[SUCCESS] Training data successfully loaded from numpy."
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

print ""
"""
print "[INFO] Loading the MNIST dataset."
digits = load_digits()
X = digits.data
y = digits.target

print "[INFO] Training the SVM model on loaded data."
""" Initialize the model """
#clf = SVC(kernel="rbf", C=2.8, gamma=.0073)
clf = SVC(kernel="rbf", C=3, gamma=.001)
clf.fit(X, y)

print "[OUTPUT] Scores on cross-validation."
""" Train and validate the model with 7-fold cross validation """
scores = cross_validation.cross_val_score(clf, X, y, cv=7)

print scores
