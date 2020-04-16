import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics 
from scipy.spatial.distance import cdist 
from scipy import cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    #Read in raw data, casting to float
    carsDF = pd.read_csv('carsDataset.csv')
    bankLoanDF = pd.read_csv('bankloan.csv')

    #Print preview of dataframe
    print("\n1. Checking and Reading the Data\n\nCars Data")
    print(carsDF.head())
    print('\n\nBank Loan Data')
    print(bankLoanDF.head())

    #Problem 1, find outliers for mpg, qsec and hp for cars data
    print("\n2. Calculating outliers for target columns, using anything above/below 2 stdv's.")
    factor = 2
    Cols = ['mpg', 'qsec', 'hp']
    outliers = {}
    for col in Cols:
        upper_lim = carsDF[col].mean () + carsDF[col].std () * factor
        lower_lim = carsDF[col].mean () - carsDF[col].std () * factor
        outliers[col] = carsDF[(carsDF[col] >= upper_lim) | (carsDF[col] <= lower_lim)]
        print("\n\nThe outliers for {0} are:\n{1}".format(col, outliers[col]))

    #Problem 2, make boxlplots for target cols in bankloan data    
    print("\n3. Generating boxplots for bank loan data")
    Cols = ['x1', 'x5', 'x6', 'x7', 'x11', 'x13', 'x14']
    for col in Cols:
        boxplot = bankLoanDF[col].to_frame().dropna().boxplot()
        plt.title("Boxplot for {0}".format(col))
        plt.show()

    print("\n3. Getting top 10 outliers from clusters")
    #scale the data to calculate distances
    scaler = MinMaxScaler()
    bankLoanSeries = scaler.fit_transform(bankLoanDF.fillna(0))
    #Create clusters from dataset
    outlier_detection = DBSCAN(min_samples = 2, eps = 1.13)
    clusters = outlier_detection.fit_predict(bankLoanSeries)
    #Trim dataframe to outliers as identified by cluster analysis
    outliers = bankLoanDF.iloc[(clusters == -1).nonzero()]
    print(outliers)