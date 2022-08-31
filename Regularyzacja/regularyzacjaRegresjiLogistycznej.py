import os
import pandas as pd
from matplotlib import pyplot
from numpy import linspace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

path = os.getcwd() + '/breast_cancer.txt'
names = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
         'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
dataset = pd.read_csv(path, header=None, names=names)

dataset['Class'].replace(2, 0, inplace=True)
dataset['Class'].replace(4, 1, inplace=True)

# sprawdzenie czy wystepuja brakujace wartosci i uzupelnienie ich mediana
print(dataset.isnull())
for name in names:
    dataset[name].fillna(dataset[name].median(), inplace=True)

# podzial na cechy i etykiete
X = dataset[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
             'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']]
y = dataset['Class']

# podzial zbioru na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model regresji logistycznej
model = LogisticRegression(penalty='l1', C=1, solver='liblinear', multi_class='auto')
model.fit(X_train, y_train)

print("Skutecznosc treningowa:", model.score(X_train, y_train))
print("Skutecznosc testowa:", model.score(X_test, y_test))

# wykres zaleznosci skutecznosci algorytmu wzgledem sciezki regularyzacji L2 od parametru C
effectiveness_train = []
effectiveness_test = []
C_values = linspace(0.0001, 1, 10)
for c in C_values:
    model = LogisticRegression(penalty='l2', C=c)
    model.fit(X_train, y_train)
    effectiveness_train.append(model.score(X_train, y_train))
    effectiveness_test.append(model.score(X_test, y_test))

pyplot.plot(C_values, effectiveness_train, C_values, effectiveness_test)
pyplot.xlabel('C')
pyplot.ylabel('effectiveness')
pyplot.legend(['train', 'test'])
pyplot.show()




