#color,director_name,num_critic_for_reviews,duration,director_facebook_likes,actor_3_facebook_likes,actor_2_name,actor_1_facebook_likes,gross,genres,actor_1_name,movie_title,num_voted_users,cast_total_facebook_likes,actor_3_name,facenumber_in_poster,plot_keywords,movie_imdb_link,num_user_for_reviews,language,country,content_rating,budget,title_year,actor_2_facebook_likes,imdb_score,aspect_ratio,movie_facebook_likes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helpers
import preprocess
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from visualization import *
from predictors import *
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', header=0)
ids = [
    'movie_facebook_likes',
    'duration',
    'director_facebook_likes',
    'actor_3_facebook_likes',
    'actor_2_facebook_likes',
    'actor_1_facebook_likes',
    'facenumber_in_poster',
    'budget',
    'imdb_score'
    ]

def visualize_data ():
    imdb_hist.plot_data(df)
    imdb_budget_scatter.plot_data(df)
    imdb_castlikes_scatter.plot_data(df)
    imdb_directorlikes_scatter.plot_data(df)
    imdb_country_box.plot_data(df)

def plot_correlation_matrix (data):
    correlation_matrix = data.corr()
    plt.imshow(correlation_matrix, cmap=plt.cm.Spectral_r, interpolation='nearest')
    plt.xticks(range(len(correlation_matrix)), ids, fontsize=10, rotation='vertical')
    plt.yticks(range(len(correlation_matrix)), ids, fontsize=10)
    plt.colorbar()
    plt.show()

data = df[ids]
# plot_correlation_matrix(data)

X, Y = helpers.get_numpy_data(data)
X_normed = preprocessing.normalize(X, axis=0)
X_scaled = preprocessing.scale(X_normed)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2)

# y_test = helpers.quantize_scores(y_test)
# y_train = helpers.quantize_scores(y_train)
# model = LogisticRegression()
# model = model.fit(x_train, y_train)
# print model.predict(x_test)
# print y_test
# print model.score(x_test, y_test)

# clf = decision_tree.train(x_train, y_train)
# print decision_tree.test(clf, x_test, y_test)
#
l = linear_regression.train(x_train, y_train)
print linear_regression.test(l, x_test, y_test)
#
# clf = naive_bayes.train(x_train, y_train)
# print naive_bayes.test(clf, x_test, y_test)

# clf = svm.train(x_train, y_train)
# print svm.test(clf, x_test, y_test)

clf = random_forest.train(x_train, y_train)
print random_forest.test(clf, x_test, y_test)
