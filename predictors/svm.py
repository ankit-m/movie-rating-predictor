from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

def train (x, y):
    clf = svm.SVR(kernel='rbf')
    return clf.fit(x, y)

def plot_predictions (h, y):
    plt.plot(h, color='r', linestyle=':', marker='o')
    plt.plot(y, color='b', linestyle=':', marker='o')
    plt.show()

def test (clf, x, y):
    h = clf.predict(x)
    # plot_predictions(h, y)
    return np.mean(abs(np.array(h) - np.array(y)))/10
