import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
import csv

df = pd.read_csv('../data/wind_speeds.csv')
print(df.dtypes)
X = df.loc[:, df.columns != 'farm_type']
print(X)
print()
print('--------------------------------------------------------------------------------')
print()

bkmeans = BisectingKMeans(n_clusters = 100, init='k-means++')

bkmeans.fit(X[X.columns[0:2]])

X['cluster_label'] = bkmeans.fit_predict(X[X.columns[0:2]])

centers = bkmeans.cluster_centers_

labels = bkmeans.predict(X[X.columns[0:2]])

print(X.head(10))

X.to_csv('../data/cluster_output.csv', index=None, header=True)

X.plot.scatter(x = 'lat', y = 'long', c=labels, s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()