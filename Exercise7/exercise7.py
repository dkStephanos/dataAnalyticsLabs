import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA

#Read in .dat file
intrntAddictionDF = pd.read_csv('Nichols & Nicki (2004).dat',
                 sep="\s+", #separator whitespace
                 index_col=0,
                 header=None)

#Add index column and set first row as headers
intrntAddictionDF = intrntAddictionDF.reset_index()
intrntAddictionDF.columns = intrntAddictionDF.iloc[0]
intrntAddictionDF = intrntAddictionDF[1:].astype(str).astype(int)

#Print preview of dataframe
print("\n1. Checking and Reading the Data\n")
print(intrntAddictionDF)

#Calculate and print correlation matrix
print("\n2. Checking the correlation matrix\n")
corrMatrix = intrntAddictionDF[1:].corr().round(2)
print(corrMatrix)

#Calculate and print correlation matrix means
print("\n3. Cols with corr means below 30%, marked for removal\n")
corrMeans = corrMatrix.mean()
print(corrMeans.where(corrMeans < .3).dropna().round(2).to_string(header = False))

#Drop cols with low corr corrMeans, we can do this because a low correlation indicates
#that these cols don't contribute much to the overall dataset
intrntAddictionDF = intrntAddictionDF.drop(columns=['ias13', 'ias22', 'ias32'])

#Calculate and print column variances
print("\n4. Cols with variance is below 20%, marked for removal\n")
varMatrix = intrntAddictionDF.var()
print(varMatrix.where(varMatrix < .2).dropna().round(2).to_string(header = False))

#Drop cols with low variance, we should do this because these cols dont vary much
#which means they don't impact the dataset, and likely reflect an irrelevant question
intrntAddictionDF = intrntAddictionDF.drop(columns=['ias23', 'ias34'])

#Make a PCA Model, and train on our data set, and print the number of
#principal components that we need to define our data with a 90% variance threshold
pca = PCA(.9)
pca.fit(intrntAddictionDF)
print("\n5. Principal Components required to describe data:\n")
print(pca.components_)

#Our model found that a total of 17 components are required to represent 90% of the
#variance of our origninal data set. Using the kaiser rule, we should only keep the first
#4 components and drop the rest, which would result in ~60% of our original dataset
#variance being reflected in those 4 principal components
