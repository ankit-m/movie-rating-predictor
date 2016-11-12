import numpy as np

def get_numpy_data (df):
    numeric_df = df._get_numeric_data()
    scores = numeric_df['imdb_score']
    del numeric_df['imdb_score']
    nparray = numeric_df.as_matrix()
    scores_array = scores.tolist()
    return (nparray, scores_array)

def get_unique_cols (df):
    return df[[
        'movie_facebook_likes',
        'duration',
        'director_facebook_likes',
        'actor_3_facebook_likes',
        'actor_2_facebook_likes',
        'actor_1_facebook_likes',
        # 'facenumber_in_poster',
        'budget',
        # 'title_year',
        'imdb_score'
    ]]

def partition_data (df):
    X, Y = get_numpy_data(df)
    x = np.nan_to_num(X)
    return (x, Y)

def quantize_scores (Y):
    y = []
    for i in Y:
        y.append(int(round(i)))
    return y

def normalize_data (X):
    X_normed = X / X.max(axis=0)
    return X_normed
