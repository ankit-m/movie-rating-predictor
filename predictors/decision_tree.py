from sklearn import tree

def train (x, y):
    clf = tree.DecisionTreeClassifier()
    return clf.fit(x, y)

def test (clf, x, y):
    return clf.score(x, y)
