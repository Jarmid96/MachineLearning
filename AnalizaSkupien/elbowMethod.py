import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=210, centers=5, n_features=2, cluster_std=0.8, shuffle=True,random_state=0)
plt.scatter(X[:,0], X[:,1], c='red', marker='x')
plt.show()

def elbow(X, max_cluster_nr, inflection_threshold):
    inertia_km = []
    cluster_number = range(1, max_cluster_nr)
    for cl_nr in cluster_number:
        km = sklearn.cluster.KMeans(n_clusters=cl_nr, init='random', n_init=10, max_iter=300, tol=1e-4,
                                    random_state=None)
        predict = km.fit_predict(X)
        inertia_km.append(km.inertia_)
    for i in range(1, len(inertia_km) - 1):
        if inertia_km[i + 1] / inertia_km[i] > inflection_threshold:
            best_cluster_nr = i
            break
    plt.plot(cluster_number, inertia_km, best_cluster_nr, inertia_km[best_cluster_nr - 1], 'ro')
    plt.xlabel('cluster_nr')
    plt.ylabel('inertia')
    plt.show()
    return best_cluster_nr, inertia_km[best_cluster_nr-1], predict, km;

print("Punkt przegięcia (optymalna liczba klastrow): ", elbow(X, 10, 0.9)[0])