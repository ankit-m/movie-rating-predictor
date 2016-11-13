from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import helpers

classifiers = {
    'decision_tree': tree.DecisionTreeClassifier(max_depth = 6),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(max_depth=6, n_estimators=100),
    'neural_network': MLPClassifier(alpha=1),
    'svm': SVC(gamma=2, C=1),
    'ada_boost': AdaBoostClassifier()
}

def run (X, Y):
    for i in classifiers:
        scores = []
        for j in range(10):     # 10 fold cross validation
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            x_train, x_test = helpers.scale_data(x_train, x_test)
            y_train = helpers.quantize_scores(y_train)
            y_test = helpers.quantize_scores(y_test)
            classifiers[i].fit(x_train, y_train)
            scores.append(classifiers[i].score(x_test, y_test))
        print i,':', sum(scores)/float(len(scores))*100
