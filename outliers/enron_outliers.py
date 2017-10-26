#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
del data_dict["TOTAL"]
print data_dict
features = ["salary", "bonus"]

data = featureFormat(data_dict, features)

print len(data)
### your code below




for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus,color="blue" )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
