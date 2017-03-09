import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from ScikitLearn.data import plot_decision_regions, X_train, X_test, y_train, y_test

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal lenght [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

