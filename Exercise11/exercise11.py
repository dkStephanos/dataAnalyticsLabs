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
