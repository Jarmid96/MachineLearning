import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from numpy import linspace
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# wczytywanie danych
dane = loadmat('mnist.mat')

# Zad 1. Podziel dane na parametry X oraz odpowiedź y:
X = dane['X']
y = dane['y']

# Standaryzacja
for i in range(X.shape[0]):
    X[i, :] = X[i, :] / np.std(X[i, :])

# Zamiana cyfry 10 -> 0 (błąd w zbiorze danych)
y[np.where(y == 10)] = 0

# wysokość i szerokość obrazka z cyfrą
h = 20
w = 20

# Zad 4. Proszę podzielić zbiór danych na uczący (70 %) i treningowy.
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.3)

effectiveness_train = []
effectiveness_test = []
DEPTH_values = linspace(1, 50, 50)
DEPTH_best = 0
effectiveness_best = 0
for DEPTH in DEPTH_values:
    classifier = tree.DecisionTreeClassifier(max_depth=DEPTH)
    classifier.fit(features_train, labels_train)
    effectiveness_train.append(classifier.score(features_train, labels_train))
    effectiveness_test.append(classifier.score(features_test, labels_test))
    if classifier.score(features_test, labels_test) > effectiveness_best:
        effectiveness_best = classifier.score(features_test, labels_test)
        DEPTH_best = DEPTH

pyplot.plot(DEPTH_values, effectiveness_train, DEPTH_values, effectiveness_test)
pyplot.xlabel('depth')
pyplot.ylabel('effectiveness')
pyplot.legend(['train', 'test'])
pyplot.show()
print('Najlepsze glebokosc drzewa: ', DEPTH_best)
