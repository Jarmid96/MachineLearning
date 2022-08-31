from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


x, y = make_moons(n_samples=200, noise=.05, random_state=0)
plt.scatter(x[:,0],x[:,1])
plt.show()
n_clusters = 2

# DBSCAN
plt.figure()
cluster = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
cluster_labels = cluster.fit_predict(x)
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(x[:, 0], x[:, 1], marker='.', s=50, lw=0, alpha=0.7, c=colors, edgecolor='k')
plt.title('DBSCAN')
plt.show()

# KMeans
cluster = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = cluster.fit_predict(x)

plt.figure()
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(x[:, 0], x[:, 1], marker='.', s=50, lw=0, alpha=0.7, c=colors, edgecolor='k')
plt.title('KMeans')

# Agglomerative
linkage = ['ward', 'complete', 'average', 'single']
for i in linkage:
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=i)
    cluster_labels = cluster.fit(x)
    plt.figure()
    colors = cm.nipy_spectral(cluster_labels.labels_.astype(float) / n_clusters)
    plt.scatter(x[:, 0], x[:, 1], marker='.', s=50, lw=0, alpha=0.7, c=colors, edgecolor='k')
    plt.title('agglomerative clustering (linkage - ' + i + ')')
plt.show()