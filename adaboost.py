from sklearn.externals.six.moves import zip

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import numpy as np

X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
                               n_classes=3, random_state=1)

n_split = 3000

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]

print(X_train.shape)
print(y_train.shape)

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_real1 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

bdt_real.fit(X_train, y_train)
bdt_real1.fit(X_train, y_train)
# bdt_discrete.fit(X_train, y_train)

n_trees_real = len(bdt_real)
n_trees_real1 = len(bdt_real1)
# n_trees_discrete = len(bdt_discrete)

real_score = bdt_real.score(X_test, y_test)
real1_score = bdt_real1.score(X_test, y_test)
# discrete_score = bdt_discrete.score(X_test, y_test)
print("Real Score = {}".format(real_score))
print("Real1 Score = {}".format(real1_score))
# print("Discrete Score = {}".format(discrete_score))

i = 0
# for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
for real_test_predict, real1_test_predict in zip(bdt_real.staged_predict(X_test), bdt_real1.staged_predict(X_test)):
    print("{}".format(i), end="\t")
    print("Real = {}".format(accuracy_score(real_test_predict, y_test)), end="\t")
    print("Real1 = {}".format(accuracy_score(real1_test_predict, y_test)), end="\n")
    # print("Discrete = {}".format(accuracy_score(discrete_train_predict, y_test)))
    i += 1

real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
real1_estimator_errors = bdt_real1.estimator_errors_[:n_trees_real1]
print("Real Est Err = {}".format(np.mean(real_estimator_errors)))
print("Real1 Est Err = {}".format(np.mean(real1_estimator_errors)))
# discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
