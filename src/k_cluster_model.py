import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

df = pd.read_csv('../data/wind_speeds.csv')
X = df.loc[:, df.columns != 'farm_type']
print(X)
print()
print('--------------------------------------------------------------------------------')
print()

# K_clusters = range(1,10)

# kmeans = [KMeans(n_clusters=i, n_init='auto') for i in K_clusters]

# Y_axis = X[['lat']]
# X_axis = X[['long']]

# score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

# plt.plot(K_clusters, score)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve')
# plt.show()

kmeans = KMeans(n_clusters = 4, init='k-means++', n_init='auto')

kmeans.fit(X[X.columns[0:2]])

X['cluster_label'] = kmeans.fit_predict(X[X.columns[0:2]])

centers = kmeans.cluster_centers_

labels = kmeans.predict(X[X.columns[0:2]])

print(X.head(10))

X.plot.scatter(x = 'lat', y = 'long', c=labels, s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()