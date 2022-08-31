import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering

path = os.getcwd() + '/shopping_data.csv'
customer_data = pd.read_csv(path)
data = customer_data.iloc[:, 3:5].values

n_clusters = 5
linkage = ['ward', 'complete', 'average', 'single']
for i in linkage:
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=i)
    cluster_labels = cluster.fit_predict(data)

    plt.figure()
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    plt.scatter(data[:, 0], data[:, 1], marker='.', s=50, lw=0, alpha=0.7, c=colors, edgecolor='k')
    plt.title(i)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')

plt.show()