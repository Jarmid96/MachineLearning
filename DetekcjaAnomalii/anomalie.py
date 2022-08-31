import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
from sklearn.metrics import f1_score


def estimate_gaussian(X):
    mu = []
    sigma2 = []
    for i in range(X.shape[1]):
        mu.append(np.mean(X[:, i]))
        sigma2.append(np.var(X[:, i]))
    return mu, sigma2


def select_threshold(pval, yval):
    epsilon = []
    f1 = []

    eps_table = np.linspace(pval.min(), pval.max(), 1000)
    for eps in eps_table:
        pred = np.where((pval[:, 0] < eps), (np.where((pval[:, 1] < eps), 1, 1)), (np.where((pval[:, 1] < eps), 1, 0)))
        epsilon.append(eps)
        f1.append(f1_score(yval, pred))

    best_f1 = np.max(f1)
    best_epsilon = epsilon[np.argmax(f1)]
    return best_epsilon, best_f1


data = loadmat('ex8data1.mat')
X = data['X']

# amount of data
print('X shape: ', X.shape)

# visualise the dataset and look for anomalies
x = X[:, 0]
y = X[:, 1]
plt.figure()
plt.scatter(x, y)
plt.show()

# plot the histogram for features: throughput (mb/s) and latency (ms)
plt.figure()
plt.subplot(1,2,1), plt.hist(x, bins=50), plt.title('throughput feature histogram')
plt.subplot(1,2,2), plt.hist(y, bins=50), plt.title('latency feature histogram')
plt.show()

# calculate mu and sigma
mu, sigma2 = estimate_gaussian(X)
print(mu, sigma2)

Xval = data['Xval']
yval = data['yval']

# check the number of data (X.shape)
print('Xval shape: ', Xval.shape)
print('yval shape: ', yval.shape)

# calculate the probability for X data
# calculate the distance
p = np.zeros((X.shape[0], X.shape[1]))
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
p[:, 0] = stats.norm.pdf(X[:, 0], mu[0], np.sqrt(sigma2[0]))
p[:, 1] = stats.norm.pdf(X[:, 1], mu[0], np.sqrt(sigma2[0]))
pval[:, 0] = stats.norm.pdf(Xval[:, 0], mu[0], np.sqrt(sigma2[0]))
pval[:, 1] = stats.norm.pdf(Xval[:, 1], mu[1], np.sqrt(sigma2[1]))

best_epsilon, best_f1 = select_threshold(pval, yval)
print(best_epsilon, best_f1)

# find indexes where the p value is lower than epsilon. Use the np.where() function
indexes = np.where(pval < best_epsilon)[0]

# plot the data and analyse the outcome. Use plt.scatter() function
plt.figure()
x = Xval[:, 0]
y = Xval[:, 1]
xx = Xval[indexes[0]:indexes[len(indexes)-1], 0]
yy = Xval[indexes[0]:indexes[len(indexes)-1], 1]
plt.scatter(x, y)
plt.scatter(xx, yy, s=200, facecolors='none', edgecolors='r')
plt.show()