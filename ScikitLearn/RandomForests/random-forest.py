import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ScikitLearn.data import plot_decision_regions, X_train, y_train, X_combined, y_combined

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=2,
                                random_state=1,
                                n_jobs=2
                                )

forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()