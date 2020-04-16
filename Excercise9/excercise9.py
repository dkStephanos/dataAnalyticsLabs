import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    potentialClientsDF = pd.read_csv('PotentialClients.csv')

    #Print preview of dataframe
    print("\n1. Checking and Reading the Data\n")
    print(potentialClientsDF.head())

    #Plotting raw data
    plt.plot(potentialClientsDF, 'x')
    plt.show()