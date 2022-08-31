import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.cluster.hierarchy as ch

path = os.getcwd() + '/shopping_data.csv'
customer_data = pd.read_csv(path)
data = customer_data.iloc[:, 3:5].values

plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
# Proszę wyliczyć odległość dla metody Warda (funkcja linkage)
Z = ch.linkage(data, 'ward')
# Przy pomocy funkcji dendrogram wyświetl wynik
ch.dendrogram(Z)
plt.show()