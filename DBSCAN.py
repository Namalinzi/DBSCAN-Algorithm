# DBSCAN Clustering

# **Steps**
# - Import the dataset
# - Import the DBSCAN from sklearn
# - Fitting the model
#

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values # only considering features in index 3 and 4


dataset

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=3,min_samples=4)

# Fitting the model

model=dbscan.fit(X)

labels=model.labels_ # this gives all the labels obtained


labels # -1 indicates that those points are outliers and thus not included in any clusters

from sklearn import metrics

#identifying the points which makes up our core points 
sample_cores=np.zeros_like(labels,dtype=bool)

sample_cores[dbscan.core_sample_indices_]=True

# Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0)

n_clusters

print(metrics.silhouette_score(X,labels))



