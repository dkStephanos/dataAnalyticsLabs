import csv
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
#from sklearn.linear_model import LinearRegression
#from sklearn import metrics
#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

albumSalesDF = pd.read_csv('Album Sales 3.csv')

#Print preview of dataframe
print("1. Checking and Reading the Data\n\n")
print(albumSalesDF)

# Set contants for scatter plots
colors = np.array([0.5, 0.5, 0.5])
area = np.pi*3

# Plot Sales and Adverts
plt.scatter(albumSalesDF.adverts, albumSalesDF.sales, s=area, c=colors, alpha=0.5)
plt.title('Sales v Adverts')
plt.xlabel('Adverts')
plt.ylabel('Sales')
#plt.show()

# Plot Sales and Airplay
plt.scatter(albumSalesDF.airplay, albumSalesDF.sales, s=area, c=colors, alpha=0.5)
plt.title('Sales v Airplay')
plt.xlabel('Airplay')
plt.ylabel('Sales')
#plt.show()

# Plot Sales and Attract
plt.scatter(albumSalesDF.attract, albumSalesDF.sales, s=area, c=colors, alpha=0.5)
plt.title('Sales v Attract')
plt.xlabel('Attract')
plt.ylabel('Sales')
#plt.show()

# Single Variable Linear Regression
# create a fitted model
lm1 = smf.ols(formula='sales ~ adverts', data=albumSalesDF).fit()

# print the models params, F-Statistics and p-values
print("\nLinear Regression p-values & summary")
print(lm1.pvalues.to_string())
print(lm1.summary())
print("\n\nHere the p-value is an indication of statistical signifigance, which is to say that our results" +
" are meaningful provided the p-value is less than 5%, which it is. The F-statistic is a measure of how well our " +
 "data fits to the model. The higher the better, so this is another indication that our model is pretty good.")

# print the Linear Regression Model Params and estimate for $135,000 in advert
print("\nLinear Regression Model Params")
print(lm1.params.to_string())
print("\nEstimate for $135,000 in advert")
print(lm1.params[0] + (lm1.params[1] * 135000))


# create a fitted model with all three features
lm2 = smf.ols(formula='sales ~ adverts + airplay + attract', data=albumSalesDF).fit()

# print the coefficients
print("\nMulti-Variable Linear Regression p-values & summary")
print(lm2.pvalues.to_string())
print(lm2.summary())
print("\n\nBased on this summary, our p-value still indicates signifigance, but our F-statistic and Rsquared value are much higher"
+ " which means that the multiple variable regression provides a better hypothesis for the data set, which is to say that, " +
"when calculating a price prediction, this model will provide a more accurate estimation than that of the single variable model")
