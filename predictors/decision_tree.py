from sklearn import tree
import helpers

def train (x, y):
    y = helpers.quantize_scores(y)
    clf = tree.DecisionTreeClassifier()
    return clf.fit(x, y)

def test (clf, x, y):
    y = helpers.quantize_scores(y)
    return clf.score(x, y)
