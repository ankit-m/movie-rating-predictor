from sklearn.naive_bayes import GaussianNB
import helpers

def train (x, y):
    y = helpers.quantize_scores(y)
    clf = GaussianNB()
    return clf.fit(x, y)

def test (clf, x, y):
    y = helpers.quantize_scores(y)
    print clf.predict(x), y
    return clf.score(x, y)
