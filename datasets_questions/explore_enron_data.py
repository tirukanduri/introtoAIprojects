#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data["SKILLING JEFFREY K"])
count = 0
for key in enron_data:
    if(enron_data[key]["poi"])==1:
        count+=1

print count
print "SKILLING JEFFREY K"
print enron_data["SKILLING JEFFREY K"]["total_payments"]

print "LAY KENNETH L"
print enron_data["LAY KENNETH L"]["total_payments"]

print "FASTOW ANDREW S"
print enron_data["KAMINSKI WINCENTY J"]["deferral_payments"]

salCount=0
emailCount=0
noSalCount=0
counter = 0
poiCountWithnoPayments=0;
poiCount=0
for salKey in enron_data:
    if(enron_data[salKey]["salary"]!="NaN"):
        salCount+=1

    if(enron_data[salKey]["email_address"]!="NaN"):
        emailCount+=1
    if    rdd(enron_data[salKey]["total_payments"]=="NaN"):
        noSalCount+=1

    if (enron_data[key]["poi"]) == 1:
        poiCount+=1
        if(enron_data[key]["total_payments"]=="NaN"):
            poiCountWithnoPayments += 1

    counter +=1

print(salCount)
print(emailCount)

print(noSalCount)
print(noSalCount*100/counter)

print(poiCount)
print(poiCountWithnoPayments)
print ((poiCountWithnoPayments+10)*100/(poiCount+10))

print counter


