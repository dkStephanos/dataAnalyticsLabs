import csv
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
# row data [id, gender, age, income, tax(15%)]
incomeData = pd.read_csv('Income Dirty Data.csv', index_col=0)

print("1. Checking and Reading the Data\n\n")
print(incomeData)

print("\nTotal NaN count for each column:")
isNaNperColumn = incomeData.isna().sum()
print(isNaNperColumn.to_string())

print("\nPercentage of NaN values:")
print(isNaNperColumn.sum()/(len(incomeData)*len(incomeData.columns)))

print("\nNumber of rows without Nan values:")
rowsWithNaN = len(incomeData[(incomeData['gender'].isna()) | (incomeData['income'].isna()) | (incomeData['tax (15%)'].isna())])
print(len(incomeData)-rowsWithNaN)

print("\n\n2. Checking with Rules\n")
#Rules:
#Age must be over 18
#Income must be positive
#Income tax must be 15% of income
print("\nNumber of rows that pass rules:")
validRows = len(incomeData[(incomeData['age'] > 18) & (incomeData['income'] > 0) & (incomeData['tax (15%)'] == incomeData['income']*.15)])
print(validRows)

print("\nPercentage of rows that pass rules:")
print(validRows/len(incomeData))

print("\n\n3. Correcting the data\n")
#Converting NA in gender column to either Male or Female randomly
incomeData['gender'] = incomeData['gender'].fillna(pd.Series(np.random.choice(['Male', 'Female'], size=len(incomeData.index))))

#Replaces gender entries (Man, Men) and (Woman, Women) with Male, and Female respectively
incomeData['gender'] = incomeData['gender'].apply(lambda gender: 'Male' if (gender == 'Man' or gender == 'Men') else ('Female' if (gender == 'Women' or gender == 'Woman') else gender))

#Converts all negative values to NaN
incomeData.loc[~(incomeData['age'] > 0), 'age']=np.nan
incomeData.income[incomeData.income < 0] = np.NaN
incomeData["tax (15%)"][incomeData["tax (15%)"] < 0] = np.NaN

#If income is NaN, derive it from tax
incomeData['income'] = np.where((incomeData.income.isna() & incomeData['tax (15%)'].notnull()),incomeData['tax (15%)']*(100/15),incomeData.income)

#If tax is NaN, derive it from income
incomeData['tax (15%)'] = np.where((incomeData.income.notnull() & incomeData['tax (15%)'].isna()),incomeData.income*.15,incomeData['tax (15%)'])

#Prints cleaned incomeData and final total and percentage of rows that pass rules
minValuesObj = incomeData.min()
print('\nminimum value in each column: ')
print(minValuesObj[1:].to_string())

maxValuesObj = incomeData.max()
print('\nmaximum value in each column: ')
print(maxValuesObj[1:].to_string())

meanValuesObj = incomeData.mean()

print('\nmean value in each column: ')
print(meanValuesObj[1:].to_string())

medianValuesObj = incomeData.median()

print('\nmedian value in each column: ')
print(medianValuesObj[1:].to_string())

quantileValuesObj = incomeData[1:].quantile([.25, .75])

print('\n1st and 3rd quartiles for each column: ')
print(quantileValuesObj.to_string())

print("\nTotal NaN count for each column:")
isNaNperColumn = incomeData.isna().sum()
print(isNaNperColumn.to_string())

#Encode the gender column (Male, Female) to (0,1) for machine learning algorithm
#le = LabelEncoder()
#incomeData["gender"] = le.fit_transform(incomeData["gender"])

#Rescaling features
#scaler = StandardScaler()
#features = [['gender', 'age', 'income', 'tax (15%)']]
#for feature in features:
#    incomeData[feature] = scaler.fit_transform(incomeData[feature])

imputer = KNNImputer()
filledIncomeData = imputer.fit_transform(incomeData)

id_col = list(range(1,1000))
cols = ['gender', 'age', 'income', 'tax (15%)']

finalIncomeData = pd.DataFrame(data=filled, index=id_col, columns=cols)
finalIncomeData.index.name = "ID"

#Prints incomeData after kNN algorithm
minValuesObj = finalIncomeData.min()
print('\nminimum value in each column: ')
print(minValuesObj[1:].to_string())

maxValuesObj = finalIncomeData.max()
print('\nmaximum value in each column: ')
print(maxValuesObj[1:].to_string())

meanValuesObj = finalIncomeData.mean()

print('\nmean value in each column: ')
print(meanValuesObj[1:].to_string())

medianValuesObj = finalIncomeData.median()

print('\nmedian value in each column: ')
print(medianValuesObj[1:].to_string())

quantileValuesObj = finalIncomeData[1:].quantile([.25, .75])

print('\n1st and 3rd quartiles for each column: ')
print(quantileValuesObj.to_string())
