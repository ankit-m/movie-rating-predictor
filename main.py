#color,director_name,num_critic_for_reviews,duration,director_facebook_likes,actor_3_facebook_likes,actor_2_name,actor_1_facebook_likes,gross,genres,actor_1_name,movie_title,num_voted_users,cast_total_facebook_likes,actor_3_name,facenumber_in_poster,plot_keywords,movie_imdb_link,num_user_for_reviews,language,country,content_rating,budget,title_year,actor_2_facebook_likes,imdb_score,aspect_ratio,movie_facebook_likes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helpers
import feature_importance
from sklearn.linear_model import LogisticRegression
from visualization import *
from predictors import *
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
data = df[ids]      # features with continuous variables

def visualize_data ():
    imdb_hist.plot_data(df)
    imdb_budget_scatter.plot_data(df)
    imdb_castlikes_scatter.plot_data(df)
    imdb_directorlikes_scatter.plot_data(df)
    imdb_country_box.plot_data(df)
    correlation_matrix.plot_data(df, ids)

# visualize_data()
X, Y = helpers.get_numpy_data(data)
# classification.run(X, Y)
# regression.run(X, Y)
feature_importance.run(X, Y, ids)
