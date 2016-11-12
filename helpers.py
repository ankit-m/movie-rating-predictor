import numpy as np

def get_numpy_data (df):
    numeric_df = df._get_numeric_data()
    scores = numeric_df['imdb_score']
    del numeric_df['imdb_score']
    nparray = numeric_df.as_matrix()
    nparray = np.nan_to_num(nparray)
    scores_array = scores.tolist()
    return (nparray, scores_array)

def quantize_scores (Y):
    y = []
    for i in Y:
        y.append(int(round(i)))
    return y

def normalize_data (X):
    X_normed = X / X.max(axis=0)
    return X_normed
