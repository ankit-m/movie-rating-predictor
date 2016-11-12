from sklearn import preprocessing

def normalize():
    pass

def categorize(X):
    le = preprocessing.LabelEncoder()
    for i in X:
        # print i
        if isinstance(X[i][1], basestring) == True:
            # print i
            le.fit(X[i].values)
            print i, len(list(le.classes_))
            # print le.transform(X[i].values)
    # le.fit()

def run(X):
    categorize(X)
    # normalize(X)
