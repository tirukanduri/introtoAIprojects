import matplotlib.pyplot as plt

def drawGraph(data,xID,yID, xLabel,yLabel):

    for point in data:
        x=point[xID]
        y=point[yID]
        plt.scatter(x, y, color="blue")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()