from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
iris = load_iris()

X = DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


n_neighbours = 25

predictor = KNeighborsClassifier(n_neighbours)
predictor.fit(X_train, Y_train)
res = predictor.predict(X_test)
print(accuracy_score(res, Y_test))
print(X.describe())
