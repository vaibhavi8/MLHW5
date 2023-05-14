#-------------------------------------------------------------------------
# AUTHOR: Vaibhavi Jhawar
# FILENAME: clustering.py
# SPECIFICATION: Run k-means multiple times and check which k calue maximizes the Silhouette coefficient
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics
from matplotlib import pyplot as plt

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.values.tolist() #converts an array to a list with the items, elements, or values

# print(X_training)


#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

silhouetteScore = []
kMeans = []


     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    silhouetteScore.append(silhouette_score(X_training, kmeans.labels_))
    kMeans.append(k)


#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(kMeans, silhouetteScore)
plt.xlabel("# of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None)
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]

#--> add your Python code here
labels = np.array(df.values).reshape(1, len(df.index))[0]
#Calculate and print the Homogeneity of this kmeans clustering
# print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__()) #K-Means Homogeneity Score = 0.8710901110077528