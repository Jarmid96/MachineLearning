import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split

# pobranie danych
iris = ds.load_iris()
X = iris.data
y = iris.target

# zamiana inf na nan i ewentualne usuniecie nan
np.where(X == np.inf, np.nan, X)
print('Number of nans: ', np.sum(np.sum(np.isnan(X))))

# skalowanie danych
X = pd.DataFrame(scale(X))
print('Wymiary X:', X.shape)
print('Wymiary y:', y.shape)

# podzial zbioru na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# utworzenie macierzy kowariancji
cov_matrix = np.cov(X_train.T)
print('Macierz kowariancji: \n', cov_matrix)

# wartosci i wektory wlasne macierzy kowariancji
eigvals, eigvecs = np.linalg.eig(cov_matrix)
print('Wartosci wlasne: \n', eigvals)
print('Wektory wlasne: \n', eigvecs)

# sortowanie (po polaczeniu wartosci i wektorow wlasnych)
eig = []
for i in range(4):
    eig.append(((eigvals[i]), eigvecs[:,i]))
eig.sort(key=lambda x: x[0], reverse=True)
print(eig)

suma = sum(eigvals)
var_exp = [(i/suma) for i in sorted(eigvals, reverse=True)]

plt.plot(var_exp, 'r-')
plt.plot(var_exp, 'b*')
plt.xlabel('numery wartosci wlasnych')
plt.ylabel('wspolczynniki wariancji wyjasnionej')
plt.show()




