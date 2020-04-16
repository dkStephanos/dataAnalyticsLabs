import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics 
from scipy.spatial.distance import cdist 
from scipy import cluster
from sklearn.cluster import KMeans

if __name__ == '__main__':
    #Read in raw data, casting to float
    potentialClientsDF = pd.read_csv('PotentialClients.csv')
    potentialClientsDF = potentialClientsDF.astype('float64')

    #Print preview of dataframe
    print("\n1. Checking and Reading the Data\n")
    print(potentialClientsDF.head())

    #Plotting raw data
    plt.plot(potentialClientsDF, 'x')
    plt.title("Raw Potential Client Data")
    plt.show()

    #Calculate and print empirical method result for ideal number of clusters
    print("\nEmpirical Method # Clusters:   {0}".format(math.sqrt(potentialClientsDF.shape[0]/2)))

    #Create and fit KMeans model to data for a variety of k's, collecting distortion values for each
    #plot variance for each value for 'k' between 1,10
    initial = [cluster.vq.kmeans(potentialClientsDF,i) for i in range(1,10)]
    plt.title('The Elbow Plot for Potential Clients')
    plt.plot([var for (cent,var) in initial])
    plt.show()

    cent, var = initial[3]
    #use vq() to get as assignment for each obs.
    assignment,cdist = cluster.vq.vq(potentialClientsDF,cent)
    plt.scatter(potentialClientsDF.iloc[:,0], potentialClientsDF.iloc[:,1], c=assignment)
    plt.title('The Data Partitioned into 4 Clusters')
    plt.show()

    #Elbow plot and kmeans cluster plot implementations borrowed from: https://stats.stackexchange.com/questions/9850/how-to-plot-data-output-of-clustering