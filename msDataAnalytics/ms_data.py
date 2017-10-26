
import Util
import matplotlib.pyplot as plt

def plot(data_dict):
    for key in data_dict.keys():
        #print key
        values=data_dict[key]

        for x in values:
            plt.scatter(key,x,color="blue")
    plt.xlabel("PPL")
    plt.ylabel("access")
    plt.show()

def read_content():
    filename = "../msdata/visitstest.data"
    ptr = open(filename)
    ptr = open(filename)
    data_dict = {}
    subDict={}
    key=""
    values=[]
    for x in open(filename).readlines():

    #extract and start loading data_dict

        content = x.strip("\n").split(",")
        if content[0]=="C":
            if key!="":
                data_dict[key]=values
                values=[]
            key=content[2]
            #print key
        else:
            #print content[1]
            values.append(content[1])
        #print values
    data_dict[key]=values
    print data_dict
    return data_dict

data_dict = read_content()
print len(data_dict.keys())
print data_dict
plot(data_dict)







