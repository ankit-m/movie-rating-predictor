from sklearn.ensemble import RandomForestClassifier
import helpers

def train (x, y):
    y = helpers.quantize_scores(y)
    clf = RandomForestClassifier(n_estimators = 15)
    return clf.fit(x, y)

def test (clf, x, y):
    y = helpers.quantize_scores(y)
    return clf.score(x, y)
