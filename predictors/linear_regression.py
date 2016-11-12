from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def train (x, y):
    l = LinearRegression()
    return l.fit(x, y)

def plot_predictions (h, y):
    plt.plot(h, color='r', linestyle=':', marker='o')
    plt.plot(y, color='b', linestyle=':', marker='o')
    plt.show()

def test (l, x, y):
    h = l.predict(x)
    # plot_predictions(h, y)
    return np.mean(abs(np.array(h) - np.array(y)))/10
