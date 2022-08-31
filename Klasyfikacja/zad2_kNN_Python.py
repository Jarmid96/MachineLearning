from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# pobranie danych
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

# podzial zbioru na zbiory treningowy i testowy
features_train, features_test, labels_train, labels_test = train_test_split(iris_X, iris_Y, test_size=0.3)

# liczba rozpatrywanych najblizszych sasiadow
K = 5

# uczenie oraz przewidywanie etykiet dla zbioru testowego
classifier = KNeighborsClassifier(n_neighbors=K)
classifier.fit(features_train, labels_train)
predictions = classifier.predict(features_test)

# sprawdzanie skutecznosci klasyfikatora
output = accuracy_score(labels_test, predictions)
print(output*100, '%')