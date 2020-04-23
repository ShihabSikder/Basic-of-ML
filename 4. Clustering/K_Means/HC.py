# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:41:00 2020

@author: Shihab
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using dendagram to find optimal number of cluster

import scipy.cluster.hierarchy as sch
dendragram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendragram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distances')
plt.savefig('Dendragram.png', dpi=1600)
plt.show()

#we find 5 cluster from the dendragram

#fitting HC in datasets

from sklearn.cluster import AgglomerativeClustering as ac
hc = ac(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#visualising cluster
plt.scatter(X[y_hc==0, 0],X[y_hc==0, 1], s =100, c='red', label='Cluster 1')
plt.scatter(X[y_hc==1, 0],X[y_hc==1, 1], s =100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc==2, 0],X[y_hc==2, 1], s =100, c='green', label='Cluster 3')
plt.scatter(X[y_hc==3, 0],X[y_hc==3, 1], s =100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc==4, 0],X[y_hc==4, 1], s =100, c='magenta', label='Cluster 5')

#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Cluster of Clients using HC')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.savefig('HC clustering without centroid .png', dpi=1600)
plt.show()