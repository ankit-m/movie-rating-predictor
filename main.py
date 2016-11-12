#color,director_name,num_critic_for_reviews,duration,director_facebook_likes,actor_3_facebook_likes,actor_2_name,actor_1_facebook_likes,gross,genres,actor_1_name,movie_title,num_voted_users,cast_total_facebook_likes,actor_3_name,facenumber_in_poster,plot_keywords,movie_imdb_link,num_user_for_reviews,language,country,content_rating,budget,title_year,actor_2_facebook_likes,imdb_score,aspect_ratio,movie_facebook_likes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helpers
from sklearn.cross_validation import train_test_split
from visualization import *
from predictors import *

df = pd.read_csv('data.csv', header=0)

def visualize_data ():
    imdb_hist.plot_data(df)
    imdb_budget_scatter.plot_data(df)
    imdb_castlikes_scatter.plot_data(df)
    imdb_directorlikes_scatter.plot_data(df)
    imdb_country_box.plot_data(df)

data = helpers.get_unique_cols(df)

X, Y = helpers.partition_data(data)
X_normed = helpers.normalize_data(X)
x_train, x_test, y_train, y_test = train_test_split(X_normed, Y, test_size=0.1)

clf = decision_tree.train(x_train, y_train)
print decision_tree.test(clf, x_test, y_test)

l = linear_regression.train(x_train, y_train)
print linear_regression.test(l, x_test, y_test)

clf = naive_bayes.train(x_train, y_train)
print naive_bayes.test(clf, x_test, y_test)
