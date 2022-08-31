from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score

# pobranie danych
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

# podzial zbioru na zbiory treningowy i testowy
features_train, features_test, labels_train, labels_test = train_test_split(iris_X, iris_Y, test_size=0.5)

# przewidywania etykiet dla features_test
predictions = []

# liczba rozpatrywanych najblizszych sasiadow
K = 5

# liczba etykiet
labels_quantity = 3

# uczenie oraz przewidywanie etykiet dla zbioru testowego
for i in range(0, len(features_test)):
    distances = []
    closest = []
    labels_closest = []
    labels_closest_quantity = 0

    for j in range(0, len(features_train)):
        dst = distance.euclidean(features_test[i], features_train[j])
        distances.append([dst, labels_train[j]])

    distances.sort()
    closest = distances[:K]

    for k in range(0, K):
        labels_closest.append(closest[k][1])

    for label in range(0, labels_quantity):
        if labels_closest.count(label) > labels_closest_quantity:
            labels_closest_quantity = labels_closest.count(label)
            label_max = label

    predictions.append(label_max)

# sprawdzanie skutecznosci klasyfikatora
output = accuracy_score(labels_test, predictions)
print(output*100, '%')
