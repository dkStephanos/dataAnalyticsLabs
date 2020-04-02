import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# row data [job, type, income, education, prestige]
incomeData = pd.read_csv('Income_Data.csv', index_col=0)

print(incomeData)

minValuesObj = incomeData.min()

print('\nminimum value in each column: ')
print(minValuesObj[1:].to_string())

maxValuesObj = incomeData.max()

print('\nmaximum value in each column: ')
print(maxValuesObj[1:].to_string())

meanValuesObj = incomeData.mean()

print('\nmean value in each column: ')
print(meanValuesObj.to_string())

medianValuesObj = incomeData.median()

print('\nmedian value in each column: ')
print(medianValuesObj.to_string())

quantileValuesObj = incomeData[1:].quantile([.25, .75])

print('\n1st and 3rd quartiles for each column: ')
print(quantileValuesObj.to_string())

print('\nstandard deviation for income: ')
print(incomeData["income"].std())

print('\nvariance for income: ')
print(incomeData["income"].var())

print('\ncorrelation coefficients for income & education: ')
print(incomeData.loc[:, "income":"education"].corr().to_string())

print('\np-values for income & education: ')
print(ttest_ind(incomeData['income'], incomeData['education'])[1])

#Histogram for prestige
incomeData.hist(column="prestige")

#Plot of income and education
incomeData.plot(y=["education", "income"])

plt.show()

#Output from program

# minimum value in each column:
# income       7
# education    7
# prestige     3
# 
# maximum value in each column:
# income        81
# education    100
# prestige      97
#
# mean value in each column:
# income       41.866667
# education    52.555556
# prestige     47.688889
#
# median value in each column:
# income       42.0
# education    45.0
# prestige     41.0
#
# 1st and 3rd quartiles for each column:
#       income  education  prestige
# 0.25    20.0      25.75     16.00
# 0.75    64.0      84.00     77.25
#
# standard deviation for income:
# 24.435071664980384
#
# variance for income:
# 597.0727272727273
#
# correlation coefficients for income & education:
#              income  education
# income     1.000000   0.724512
# education  0.724512   1.000000
#
# p-values for income & education:
# 0.06592789784461847
