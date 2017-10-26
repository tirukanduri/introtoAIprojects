#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = [90]
    errors=[90]
    single_set=[]
    test_tuple=[]

    for  (networth,age,pred) in zip(net_worths,ages,predictions):
        error=abs(networth-pred)
        errors.append(error)
        single_set=(age,networth,error)
        test_tuple.append(single_set)
    #print predictions

    print ("****test tuple***")
    #for x in test_tuple:
     #   print x
    #errors.sort()
    #print errors
    #print len(errors)
    print("########")

    #cleaned_data=list(zip(ages,net_worths,errors))
    #cleaned_data.sort(errors)
    #cleaned_data.sort(key=lambda tup: tup[2])
    from operator import itemgetter
    sorted_data=sorted(test_tuple,key = itemgetter(2))

    print "Printing sorted set"
    #print sorted_data
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(81)
    sorted_data.__delitem__(80)
    #for x in sorted_data:
     #   print x
    #print sorted_data

    #print len(sorted_data)

    ### your code goes here

    
    return sorted_data

