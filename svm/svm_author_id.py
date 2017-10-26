#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
#sys.path.append("../tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#using svms kernel = linear
from sklearn.svm import SVC
svm_linear_classifier = SVC(kernel="rbf", C=10000.0)
svm_linear_classifier.fit(features_train,labels_train)
svm_linear_prediction=svm_linear_classifier.predict(features_test)

from sklearn.metrics import accuracy_score

print("SVM Linear accuracy: %s" %(accuracy_score(svm_linear_prediction,labels_test)))

counter=0;
for value in svm_linear_prediction:
    if (value==1):
        counter = counter + 1

print(counter)

#########################################################


