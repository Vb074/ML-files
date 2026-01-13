#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:45:28 2026

@author: vadimbodnarenko
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Country_Risk_2019_Data.csv')

#Checking for correlation 
print(df.corr(numeric_only = True))

#Dropping Corruption since it's highly correlated with Legal
cols = ['Peace', 'Legal', 'GDP Growth']
X = df[cols]

#Standardizing
X = (X - X.mean())/X.std(ddof = 1)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, n_init = 10)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

#Choosing a number of clusters using inertia vs k plot
Ks = range(1,10)
inertia = []
for K in Ks:
    km = KMeans(n_clusters = K, n_init = 10)
    km.fit(X)
    inertia.append(km.inertia_)
    
plt.plot(Ks, inertia, color = 'b', marker = 'o')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()

#Silhouette Score
from sklearn.metrics import silhouette_score

for k in range(2,10):
    km = KMeans(n_clusters = k, n_init = 10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f'k = {k}, silhouette score = {score:.3f}')

centers = pd.DataFrame(kmeans.cluster_centers_, columns = cols)   

#Visualizing the clusters
plt.scatter(X['Peace'], X['Legal'], c = labels) 
plt.scatter(centers['Peace'], centers['Legal'], c = 'black', s = 200)
plt.show()
    
#3D plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(X['Peace'], X['Legal'], X['GDP Growth'], c = labels, cmap = 'viridis')
ax.scatter(centers['Peace'], centers['Legal'], centers['GDP Growth'], c = 'black', s = 200)

ax.view_init(elev= 20, azim= 45 , roll=0)

ax.set_xlabel('Peace')
ax.set_ylabel('Legal')
ax.set_zlabel('GDP Growth')
plt.show()

df['Clusters'] = labels

for k in range(3):
    print(f'Cluster {k}')
    print(df[df['Clusters'] == k]['Country'].values)
    print()














