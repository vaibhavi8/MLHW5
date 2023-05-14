#-------------------------------------------------------------------------
# AUTHOR: Vaibhavi Jhawar
# FILENAME: association_rule_mining.py
# SPECIFICATION: Find strong rules related to supermarket products using dataset.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hoursd
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
for index, row in df.iterrows():

    labels = {}
    transactions = []

    for val in row:
        transactions.append(val)
    
    for i in itemset:
        if i in transactions:
            labels[i] = 1
        else:
            labels[i] = 0

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below

for index in range(len(rules["antecedents"])):
    output = ""
    for ante in rules["antecedents"][index]:
        output += ante + ", "
    output = output[:-2]
    output += " -> "
    for result in rules["consequents"][index]:
        output += result + ", "
    output = output[:-2]
    print(output)
    print("Support:", rules["support"][index])
    print("Confidence:", rules["confidence"][index])
    supportCount = 0
    for transaction in encoded_vals:
        occurs = True
        for result in rules["consequents"][index]:
            if not transaction[result]:
                occurs = False
                break
        if occurs:
            supportCount += 1
    prior = supportCount / len(encoded_vals)
    print("Prior:", prior)
    print("Gain in Confidence:", 100*(rules["confidence"][index]-prior)/prior)
    print()

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
# Wine -> Cheese
# Support: 0.2698412698412698
# Confidence: 0.6159420289855072
# Prior: 0.5015873015873016
# Gain in Confidence: 22.798569069895425

# Meat -> Cheese
# Support: 0.3238095238095238
# Confidence: 0.68
# Prior: 0.5015873015873016
# Gain in Confidence: 35.569620253164565

# Cheese -> Meat
# Support: 0.3238095238095238
# Confidence: 0.6455696202531646
# Prior: 0.47619047619047616
# Gain in Confidence: 35.569620253164565

# Eggs -> Cheese
# Support: 0.2984126984126984
# Confidence: 0.6811594202898551
# Prior: 0.5015873015873016
# Gain in Confidence: 35.80077050082554

# Milk -> Cheese
# Support: 0.3047619047619048
# Confidence: 0.6075949367088608
# Prior: 0.5015873015873016
# Gain in Confidence: 21.134433584361485

# Cheese -> Milk
# Support: 0.3047619047619048
# Confidence: 0.6075949367088608
# Prior: 0.5015873015873016
# Gain in Confidence: 21.134433584361485

# Bagel -> Bread
# Support: 0.27936507936507937
# Confidence: 0.6567164179104478
# Prior: 0.5047619047619047
# Gain in Confidence: 30.10419600112645

# Eggs -> Meat
# Support: 0.26666666666666666
# Confidence: 0.6086956521739131
# Prior: 0.47619047619047616
# Gain in Confidence: 27.826086956521753

# Meat, Eggs -> Cheese
# Support: 0.21587301587301588
# Confidence: 0.8095238095238095
# Prior: 0.5015873015873016
# Gain in Confidence: 61.39240506329114

# Cheese, Eggs -> Meat
# Support: 0.21587301587301588
# Confidence: 0.723404255319149
# Prior: 0.47619047619047616
# Gain in Confidence: 51.9148936170213

# Meat, Cheese -> Eggs
# Support: 0.21587301587301588
# Confidence: 0.6666666666666666
# Prior: 0.4380952380952381
# Gain in Confidence: 52.17391304347825

# Meat, Milk -> Cheese
# Support: 0.20317460317460317
# Confidence: 0.8311688311688312
# Prior: 0.5015873015873016
# Gain in Confidence: 65.7077100115075

# Cheese, Milk -> Meat
# Support: 0.20317460317460317
# Confidence: 0.6666666666666666
# Prior: 0.47619047619047616
# Gain in Confidence: 40.0

# Meat, Cheese -> Milk
# Support: 0.20317460317460317
# Confidence: 0.6274509803921569
# Prior: 0.5015873015873016
# Gain in Confidence: 25.09307520476545