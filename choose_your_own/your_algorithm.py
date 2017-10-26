#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################

print "Initial view complete"

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

#K nearest neighbors
from sklearn import neighbors

knn_clf= neighbors.KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(features_train,labels_train)
pred=knn_clf.predict(features_test)

from sklearn.metrics import accuracy_score
knn_accuracy=accuracy_score(pred,labels_test)
print ("k nearest neighbors with accuracy is %f " %(knn_accuracy))

#Naive Bayes
from sklearn.naive_bayes import  GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(features_train,labels_train)
nb_predict=nb_clf.predict(features_test)

print ("Naive Bayes: %f" %(accuracy_score(nb_predict,labels_test)))

# Decision Tree

from sklearn import tree
dt_classifier = tree.DecisionTreeClassifier(min_samples_split= 50)
dt_classifier.fit(features_train,labels_train)
pred = dt_classifier.predict(features_test)

from sklearn.metrics import accuracy_score
acc_min_samples_split_2= accuracy_score(pred,labels_test)
print ("Decision Tree: %f" %(accuracy_score(pred,labels_test)))

# SVM
from sklearn.svm import SVC

svm_linear_classifier = SVC(kernel="rbf", C=10000.0)
svm_linear_classifier.fit(features_train,labels_train)
svm_linear_prediction=svm_linear_classifier.predict(features_test)

from sklearn.metrics import accuracy_score

print("SVM Linear accuracy: %s" %(accuracy_score(svm_linear_prediction,labels_test)))


# adaboost

from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier()
adb_classifier.fit(features_train,labels_train)
adb_pred = adb_classifier.predict(features_test)
print("Adaboost accuracy: %f" %(accuracy_score(adb_pred,labels_test)))


#random forest

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(features_train,labels_train)
rf_pred = rf_classifier.predict(features_test)
print("Random forest accuracy: %f" %(accuracy_score(rf_pred,labels_test)))

def drawPicture(clf):
    try:
        print "Before calling pretty picture"
        prettyPicture(clf, features_test, labels_test)
        print "After calling pretty picture"
    except NameError:
        pass

