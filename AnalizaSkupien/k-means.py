import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.datasets import make_blobs

# Zadanie 1 k-means
print('Zadanie 1 k-means')
X, y = make_blobs(n_samples=210, centers=3, n_features=2, cluster_std=0.5, shuffle=True,random_state=0)
plt.scatter(X[:,0], X[:,1], c='red', marker='x')
plt.show()

km = sklearn.cluster.KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=None)
km.fit_predict(X)
print('Wartosc znieksztalcen: ', km.inertia_)

plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], s=70, c=km.labels_, cmap=plt.cm.prism)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='*', s=200,
            color='blue', label='Centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

# Zadanie 2 k-means
print('Zadanie 2 k-means')
X2, y2 = make_blobs(n_samples=210, centers=5, n_features=2, cluster_std=0.8, shuffle=True,random_state=0)
plt.scatter(X2[:,0], X2[:,1], c='red', marker='x')
plt.show()

km1 = sklearn.cluster.KMeans(n_clusters=5, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=None)
km1.fit_predict(X2)
km2 = sklearn.cluster.KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=None)
km2.fit_predict(X2)
print('Wartosc znieksztalcen (n_cluster=5, cluster_std=0.8): ', km1.inertia_)
print('Wartosc znieksztalcen (n_cluster=2, cluster_std=0.8): ', km2.inertia_)

plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], s=70, c=km1.labels_, cmap=plt.cm.prism)
plt.scatter(km1.cluster_centers_[:, 0], km1.cluster_centers_[:, 1], marker='*', s=200,
            color='blue', label='Centers')
plt.legend(loc='best')
plt.title('n_clusters=5, cluster_std=0.8')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], s=70, c=km2.labels_, cmap=plt.cm.prism)
plt.scatter(km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1], marker='*', s=200,
            color='blue', label='Centers')
plt.legend(loc='best')
plt.title('n_clusters=2, cluster_std=0.8')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

# Zadanie k-means++
print('Zadanie 3 k-means++')

kmpp = sklearn.cluster.KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None)
kmpp.fit_predict(X2)
print('Wartosc znieksztalcen (k-means++): ', kmpp.inertia_)

plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], s=70, c=kmpp.labels_, cmap=plt.cm.prism)
plt.scatter(kmpp.cluster_centers_[:, 0], kmpp.cluster_centers_[:, 1], marker='*', s=200,
            color='blue', label='Centers')
plt.legend(loc='best')
plt.title('k-means++')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

inertia_km = []
inertia_kmpp = []
std = np.linspace(0.1, 10, 100)
for i in std:
    X, y = make_blobs(n_samples=210, centers=3, n_features=2, cluster_std=i, shuffle=True, random_state=0)

    km = sklearn.cluster.KMeans(n_clusters=5, init='random', n_init=1, max_iter=3, tol=1e-4, random_state=None)
    km.fit_predict(X)
    inertia_km.append(km.inertia_)

    kmpp = sklearn.cluster.KMeans(n_clusters=5, init='k-means++', n_init=1, max_iter=3, tol=1e-4, random_state=None)
    kmpp.fit_predict(X)
    inertia_kmpp.append(kmpp.inertia_)

plt.plot(std, inertia_km, 'r-', std, inertia_kmpp, 'b-')
plt.xlabel('cluster_std')
plt.ylabel('inertia')
plt.show()


