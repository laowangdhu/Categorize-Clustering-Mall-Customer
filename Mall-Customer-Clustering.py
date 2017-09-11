# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 21:31:32 2017

@author: zaghlollight
"""
#import lib and data
import matplotlib.pyplot as plt
import pandas as pd 
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values


#using the elbow method to find the optimal number of clustters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number Of Clusters')
plt.ylabel('wcss')
plt.show()    

#applying k-means to the mall data set
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y=kmeans.fit_predict(x)
#visualise the clusters 
plt.scatter(x[y==0,0],x[y==0,1],c='red',label='C1')
plt.scatter(x[y==1,0],x[y==1,1],c='blue',label='C2')
plt.scatter(x[y==2,0],x[y==2,1],c='cyan',label='C3')
plt.scatter(x[y==3,0],x[y==3,1],c='green',label='C4')
plt.scatter(x[y==4,0],x[y==4,1],c='magenta',label='C5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()