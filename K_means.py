# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:02:14 2019

@author: Koushik
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Read the dataset
data = pd.read_csv('xclara.csv')
print("Input Data and Shape")
print(data.shape)


# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax) #Calculates the Eucledian distance

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X), size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X), size=k)
C = np.array(list(zip(C_x, C_y)))
print("Initial Centroids")
print(C)

# Plotting along with the Centroids
plt.scatter(f1, f2, s=7)
plt.scatter(C_x, C_y, marker='*', c='g')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
Difference = dist(C, C_old, None)
# Loop will run till the error becomes zero
def K_Means(X,clusters,C_old,C,Difference):
      while Difference != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = np.copy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            for j in range(len(X)):
                if clusters[j] == i:
                    points = np.array(X[j])
            C[i] = np.mean(points, axis=0)
            
        Difference = dist(C, C_old, None)

K_Means(X,clusters,C_old,C,Difference) #Calling the K-means function

#Plotting the Graph
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')


