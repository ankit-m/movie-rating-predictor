from sklearn.linear_model import LinearRegression

def train (x, y):
    l = LinearRegression()
    return l.fit(x, y)

def test (l, x, y):
    return l.score(x, y)
