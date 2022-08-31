import pandas as pd
import sklearn.cluster
from sklearn.datasets import make_moons
import seaborn as sns
import matplotlib.pyplot as plt
from elbowMethod import elbow

x, y = make_moons(1000, noise=.05, random_state=0)
X_moon = pd.DataFrame(x, columns=['f1', 'f2'])

# metoda klasteryzacji
# km -model KMeans
# y_km - wynik predykcji

best_cluster_nr, inertia, y_km, km = elbow(X_moon, 10, 0.8)

# wykres
X_moon['k_means'] = y_km
sns.lmplot(data=X_moon, x='f1', y='f2', fit_reg=False, hue='k_means', palette=['#eb6c6a', '#6aeb6c']).set(
    title='Algorytm k-srednich')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='black', s=100, alpha=0.5)
plt.show()