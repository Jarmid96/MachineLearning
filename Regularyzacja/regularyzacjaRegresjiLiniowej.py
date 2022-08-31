import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import linear_model as linm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# pobranie danych
dataset = pd.read_csv('boston.csv')
X = dataset.drop('MEDV', axis=1)
y = dataset['MEDV']

# normalizacja
X_norm = (X-X.mean())/X.std()

# podzial zbioru na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)

# utworzenie i nauka modelu regresji liniowej
model = linm.LinearRegression()
model.fit(X_train, y_train)
# wyswietlenie skutecznosci
print('Blad treningowy: {}'.format(model.score(X_train, y_train)))
print('Blad testowy: {}'.format(model.score(X_test, y_test)))

# normalizacja danych
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# podzial zbioru na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)

# utworzenie i nauka modelu regresji wielomianowej
steps = [('poly', PolynomialFeatures(degree=2)), ('model', linm.LinearRegression())]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
# wyswietlenie skutecznosci
print('Blad treningowy: {}'.format(pipe.score(X_train, y_train)))
print('Blad testowy: {}'.format(pipe.score(X_test, y_test)))

# regularyzacja metoda Ridge
steps = [('poly', PolynomialFeatures(degree=2)), ('model', linm.Ridge(alpha=10))]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
# wyswietlenie skutecznosci
print('Blad treningowy: {}'.format(pipe.score(X_train, y_train)))
print('Blad testowy: {}'.format(pipe.score(X_test, y_test)))

# wykres skutecznosci od parametru alpha
effectiveness_train = []
effectiveness_test = []
alpha_values = [0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60]
for a in alpha_values:
    steps = [('poly', PolynomialFeatures(degree=2)), ('model', linm.Ridge(alpha=a))]
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    effectiveness_train.append(pipe.score(X_train, y_train))
    effectiveness_test.append(pipe.score(X_test, y_test))

pyplot.plot(alpha_values, effectiveness_train, alpha_values, effectiveness_test)
pyplot.xlabel('alpha')
pyplot.ylabel('effectiveness')
pyplot.legend(['train', 'test'])
pyplot.show()

# regularyzacja metoda Lasso
effectiveness_train = []
effectiveness_test = []
alpha_values = np.linspace(0.001, 1, 1000)
alpha_best = 0
effectiveness_best = 0
for a in alpha_values:
    steps = [('poly', PolynomialFeatures(degree=2)), ('model', linm.Lasso(alpha=a))]
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    effectiveness_train.append(pipe.score(X_train, y_train))
    effectiveness_test.append(pipe.score(X_test, y_test))
    if pipe.score(X_test, y_test) > effectiveness_best:
        effectiveness_best = pipe.score(X_test, y_test)
        alpha_best = a

pyplot.plot(alpha_values, effectiveness_train, alpha_values, effectiveness_test)
pyplot.xlabel('alpha')
pyplot.ylabel('effectiveness')
pyplot.legend(['train', 'test'])
pyplot.show()
print('Skutecznosc testowa dla metody regularyzacji Lasso z najlepszym parametrem alpha (',
      alpha_best,') wynosi: ', effectiveness_best)
