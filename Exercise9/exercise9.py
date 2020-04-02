import csv
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    adultsData = pd.read_csv('adultsData.csv')

    #Print preview of dataframe
    print("\n1. Checking and Reading the Data\n")
    print(adultsData.head())

    #Encode dataframe to be compatible with decision tree
    conditions = ['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']
    for col in adultsData:
        if col not in conditions:
            adultsData[col] = pd.get_dummies(adultsData[col])
    print(adultsData.head())
    print(adultsData.tail())

    #Set up our target column and split the dataset into 80% Training, 20% Testing
    X = adultsData.drop(['class'], axis=1)
    y = adultsData['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and Train Decision Tree classifer object
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)

	#Predict the response for test dataset
    y_pred = clf.predict(X_test)

    #Generate confusion matrix for our results
    results = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n",results)

	#Evaluating model accuracy
    print("Accuracy:\n",metrics.classification_report(y_test, y_pred))

    #Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(X_train, y_train)

    #Predict Output
    y_pred= model.predict(X_test)
    print("Predicted Value:", y_pred)

    #Generate confusion matrix for our results
    results = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n",results)

    #Evaluating model accuracy
    print("Accuracy:\n",metrics.classification_report(y_test, y_pred))