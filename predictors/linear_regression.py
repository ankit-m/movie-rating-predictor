from sklearn.linear_model import LinearRegression

def train (x, y):
    l = LinearRegression()
    return l.fit(x, y)

def test (l, x, y):
    h = l.predict(x)
    # print h, y
    tot = 0
    for i in range(len(h)):
        # print abs(h[i] - y[i])
        global tot
        if abs(h[i] - y[i]) < 1.0:
            tot += 1
        # tot += abs(h[i] - y[i])/y[i]
        print tot
    # print '**', tot/len(h)
    print len(h)
    return l.score(x, y)
