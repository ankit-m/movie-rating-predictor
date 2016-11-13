from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import helpers

def train (x, y):
    # r = LinearRegression()
    # r = Ridge(alpha = 22.0)
    # r = BayesianRidge()
    # r = tree.DecisionTreeRegressor(max_depth=6)
    r = RandomForestRegressor(n_estimators = 400)
    return r.fit(x, y)

def test (l, x, y):
    h = l.predict(x)
    print '*****', r2_score(y, h)
    helpers.plot_predictions(h, y)
    return helpers.calc_accuracy(h, y)
