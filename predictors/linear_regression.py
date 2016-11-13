from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn import tree
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def train (x, y):
    # l = LinearRegression()
    # r = Ridge(alpha = 22.0)
    # e = BayesianRidge()
    t = tree.DecisionTreeRegressor(max_depth=6)
    return t.fit(x, y)

def plot_predictions (h, y):
    plt.plot(h, color='r', linestyle=':', marker='o')
    plt.plot(y, color='b', linestyle=':', marker='o')
    plt.show()

def test (l, x, y):
    h = l.predict(x)
    plot_predictions(h, y)
    print '**', r2_score(y, h)
    return np.mean(abs(np.array(h) - np.array(y)))
