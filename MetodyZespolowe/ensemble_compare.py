import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 1
K = 5
np.set_printoptions(precision=2)

# Load the dataset.
wine = datasets.load_wine()
X, y = [wine.data, wine.target]

# Define classifiers and classifier ensembles.
clf_0 = DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=3)
clf_1 = BaggingClassifier(random_state=RANDOM_STATE, n_estimators=50, )
clf_2 = AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=50, algorithm='SAMME')
clf_3 = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=50,
                                   min_samples_leaf=3, max_depth=1, learning_rate=1, subsample=0.5)

# Compare performance of the models.
# For cv2 = integer, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used
effectiveness_0 = cross_val_score(clf_0, X, y, cv=K)
effectiveness_1 = cross_val_score(clf_1, X, y, cv=K)
effectiveness_2 = cross_val_score(clf_2, X, y, cv=K)
effectiveness_3 = cross_val_score(clf_3, X, y, cv=K)

print('Decision tree scores:\t\t', effectiveness_0, 'average:', sum(effectiveness_0)/K)
print('Bagging scores:\t\t\t\t', effectiveness_1, 'average:', sum(effectiveness_1)/K)
print('AdaBoost scores:\t\t\t', effectiveness_2, 'average:', sum(effectiveness_2)/K)
print('Gradient boosting scores:\t', effectiveness_3, 'average:', sum(effectiveness_3)/K)

# Plot OOB estimates for Gradient Boosting Classifier.
clf_3.fit(X, y)
oob = clf_3.oob_improvement_
cumsum = np.cumsum(oob)

plt.figure()
plt.plot(cumsum, 'b-')
plt.plot(cumsum, 'ro')
plt.show()
# model Gradient Boosting przestaje dawać dalszą poprawę po okolo 4 iteracji
