from sklearn import tree
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import helpers

regressors = {
    'linear': LinearRegression(),
    'linear_ridge': Ridge(alpha = 22.0),
    'linear_bayes': BayesianRidge(),
    'decision_tree': tree.DecisionTreeRegressor(max_depth=6),
    'random_forest': RandomForestRegressor(n_estimators = 100),
    'svm': SVR(kernel='rbf')
}

def calc_accuracy (h, y):
    count = 0
    for i in range(len(h)):
        if abs(h[i] - y[i]) <= 0.7:
            count += 1
    return count/float(len(h))

def calc_residuals (h, y):
    tot = 0.0
    for i in range(len(h)):
        tot += abs(h[i] - y[i])
    return tot/len(h)

def run (X, Y):
    for i in regressors:
        scores = []
        accuracies = []
        residuals = []
        for j in range(10):     # 10 fold cross validation
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            x_train, x_test = helpers.scale_data(x_train, x_test)
            h = regressors[i].fit(x_train, y_train).predict(x_test)
            # helpers.plot_predictions(h, y_test)
            scores.append(r2_score(y_test, h))
            residuals.append(calc_residuals(h, y_test))
            accuracies.append(calc_accuracy(h, y_test))
        print i,':', sum(scores)/float(len(scores)), sum(accuracies)/float(len(accuracies))*100, sum(residuals)/float(len(residuals))
