import pandas
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv('data/training-data.csv')
print(dataset.shape)
print(dataset)

print(dataset.describe())

print(dataset.head(20))
dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()
