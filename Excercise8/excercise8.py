import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from apyori import apriori

if __name__ == '__main__':
    titanicDF = pd.read_csv('TitanicData.csv')

    #Print preview of dataframe
    print("\n1. Checking and Reading the Data\n")
    print(titanicDF.head())

    #Print summary of unique fields per col
    for col in titanicDF.columns:
        print("\nUnique values for {0}:".format(col))
        for val in titanicDF[col].unique():
            print(val)
  
    # Collecting the inferred rules in a dataframe 
    association_rules = apriori(titanicDF.values, min_support=0.005, min_confidence=0.8, min_length=2)
    association_results = list(association_rules)

    filteredResults = []
    #Filtering our results to just rules that rhs is survived
    for result in association_results:
        for entry in result.ordered_statistics:
            if entry.items_add == frozenset({'Yes'}):
                filteredResults.append(entry)

    print("\nNumber of rules: {0}\n".format(len(filteredResults)))

    #Sorting by lift
    sortedResults = sorted(filteredResults, key=lambda x: x.lift, reverse=True)
    for result in sortedResults:
        print(str(result))
