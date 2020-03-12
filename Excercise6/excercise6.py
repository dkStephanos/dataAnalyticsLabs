import csv
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats.contingency import margins

def residuals(observed, expected):
    return (observed - expected) / np.sqrt(expected)

def stdres(observed, expected):
    n = observed.sum()
    rsum, csum = margins(observed)
    # With integers, the calculation
    #     csum * rsum * (n - rsum) * (n - csum)
    # might overflow, so convert rsum and csum to floating point.
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3
    return (observed - expected) / np.sqrt(v)

americanDF = pd.read_csv('Americandata.csv')

#Print preview of dataframe
print("\n1. Checking and Reading the Data\n")
print(americanDF)

stat, p, dof, expected = chi2_contingency(americanDF.iloc[:,1:].values)

print("\n2. The chi2 results\n")
print('dof=%d' % dof)
print('p=%s' % p)
print('stat=%s' % stat)
print('\nExpected Values')
print(expected)

print("\n3. The standardized residuals")
americanStdResiduals = stdres(americanDF.iloc[:,1:].values, expected)

print('\nStandardized Residuals')
print(americanStdResiduals)

print("\n4. Our analysis")
print('\nIf our expectation fit the data, we should expect a low chi squared value and a p value above .05' +
' In this case, our chi squared is very high and our p value is very low, indicating a poor fit.' +
' Additionally, our residuals and standardized residuals are wildly off what we should expect.' +
' This is likely due to the small quantities of data, that is throwing off the chi squared test.' +
' We expect standardized residuals to fall in the range -3 to 3, so our results are a red flag.' +
' The only reasonable entry is for Housewives, which is 1.47, meaning we should be most confident' +
' in predictions related to the happiness of Housewives.')
