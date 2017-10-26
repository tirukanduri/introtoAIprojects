#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print len(features_train[0])




#########################################################
### your code goes here ###
from sklearn import tree
dt_classifier = tree.DecisionTreeClassifier(min_samples_split= 40)
dt_classifier.fit(features_train,labels_train)
pred = dt_classifier.predict(features_test)

from sklearn.metrics import accuracy_score
acc_min_samples_split_2= accuracy_score(pred,labels_test)
print accuracy_score(pred,labels_test)

#dt_classifier = tree.DecisionTreeClassifier(min_samples_split= 50)
#dt_classifier.fit(features_train,labels_train)
#pred = dt_classifier.predict(features_test)

#from sklearn.metrics import accuracy_score
#acc_min_samples_split_50= accuracy_score(pred,labels_test)
#print accuracy_score(pred,labels_test)


#########################################################

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

import math

