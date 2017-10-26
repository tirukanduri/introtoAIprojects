#!/usr/bin/python

import sys
import pickle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition.pca import RandomizedPCA
from sklearn.preprocessing.imputation import Imputer

from sklearn.svm.classes import LinearSVC

import Util

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.decomposition import PCA,NMF
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','deferral_payments','total_payments',
#                 'exercised_stock_options','bonus','restricted_stock',
#                'restricted_stock_deferred','total_stock_value', 'expenses',
#                 'loan_advances','director_fees','deferred_income','long_term_incentive'] # You will need to use more features

features_list = ['poi','salary','deferral_payments','total_payments',
                 'exercised_stock_options','bonus','restricted_stock',
                'total_stock_value', 'expenses','director_fees','deferred_income','long_term_incentive']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
# TOTAL is silly to be removed.
del data_dict["TOTAL"]
del data_dict["THE TRAVEL AGENCY IN THE PARK"]
del data_dict["LOCKHART EUGENE E"]


# not removing any others. All are people only.

data = featureFormat(data_dict, features_list)

#Util.drawGraph(data,1,6,"salary","bonus")
#Util.drawGraph(data,1,3,"salary","total_payments")
#Util.drawGraph(data,1,4,"salary","total_stock_value")
#Util.drawGraph(data,3,4,"total_payments","total_stock_value")
#Util.drawGraph(data,2,4,"bonus","total_stock_value")
#Util.drawGraph(data,)


### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#Impute NaN data

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imputedData = imputer.fit_transform(features)

#from sklearn.preprocessing import MinMaxScaler
#minMax = MinMaxScaler()
#xData = minMax.fit_transform(features)
#print xData


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train,labels_test= \
    train_test_split(features,labels,test_size=0.3, random_state=42)


pca = PCA(n_components=5)
pca.fit(features_train)
pca_features_train = pca.transform(features_train)
pca_features_test = pca.transform(features_test)


# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
#clf = RandomForestClassifier()
#clf = GradientBoostingClassifier(loss='exponential',n_estimators=1000)
#clf = GradientBoostingClassifier(loss='exponential',n_estimators=100,learning_rate=0.5)
clf = DecisionTreeClassifier(min_samples_split=100,max_depth=100)
#clf = GaussianNB()
#clf = SVC(kernel='rbf')
#clf=LinearSVC()
clf.fit(pca_features_train,labels_train)
pred = clf.predict(pca_features_test)



from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix
print accuracy_score(pred,labels_test)
print precision_score(pred,labels_test)
print recall_score(pred,labels_test)
#print classification_report(labels_test,pred)
#print confusion_matrix(labels_test,pred)

print pred

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
#tester.main()

def visualize(yvalues,xvalues, x_axis_name,y_axis_name):
    import matplotlib.pyplot as plt
    for feature, target in zip(yvalues, xvalues):
        plt.scatter(feature, target, color="blue")