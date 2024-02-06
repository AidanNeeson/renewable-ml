"""K-Means Cluster Analysis on Solar Data"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('../data/wind.csv')

X = df.loc[:, df.columns[1:4]]

# Determine the number of "optimal" clusters

# K_clusters = range(1,100)
# kmeans = [KMeans(n_clusters=i, n_init='auto') for i in K_clusters]
# Y_axis = X[['lat']]
# X_axis = X[['long']]
# score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

# plt.plot(K_clusters, score)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve')
# plt.show()

# Cluster the data

n_clusters = 20
kmeans = KMeans(n_clusters, init='k-means++', n_init='auto')
kmeans.fit(X[X.columns[0:2]])
X['cluster_label'] = kmeans.fit_predict(X[X.columns[0:2]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(X[X.columns[0:2]])

X.plot.scatter(x = 'long', y = 'lat', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=200, alpha=0.5)
plt.show()