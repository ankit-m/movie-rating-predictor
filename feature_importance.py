import helpers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def run (X, Y, ids):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    x_train, x_test = helpers.scale_data(x_train, x_test)
    forest = RandomForestRegressor(n_estimators = 250)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, ids[indices[f]], importances[indices[f]]))
    labels = []
    for i in indices:
        labels.append(ids[i])

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="0.35", edgecolor="w", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), labels, rotation="vertical")
    plt.xlim([-1, X.shape[1]])
    plt.show()
