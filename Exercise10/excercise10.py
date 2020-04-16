import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans

if __name__ == '__main__':
    potentialClientsDF = pd.read_csv('PotentialClients.csv')

    #Print preview of dataframe
    print("\n1. Checking and Reading the Data\n")
    print(potentialClientsDF.head())

    #Plotting raw data
    plt.plot(potentialClientsDF, 'x')
    plt.title("Raw Potential Client Data")
    plt.show()

    #Create and fit KMeans model to data for a variety of k's
    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    K = range(1,10) 
  
    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k).fit(potentialClientsDF) 
        kmeanModel.fit(potentialClientsDF)     
      
        distortions.append(sum(np.min(cdist(potentialClientsDF, kmeanModel.cluster_centers_, 
                          'euclidean'),axis=1)) / potentialClientsDF.shape[0]) 
        inertias.append(kmeanModel.inertia_) 
  
        mapping1[k] = sum(np.min(cdist(potentialClientsDF, kmeanModel.cluster_centers_, 
                     'euclidean'),axis=1)) / potentialClientsDF.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 

    plt.plot(K, distortions, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Distortion') 
    plt.title('The Elbow Method using Distortion') 
    plt.show() 

    kmeans = KMeans(n_clusters=2, random_state=0).fit(potentialClientsDF)