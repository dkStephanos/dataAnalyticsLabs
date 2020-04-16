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

