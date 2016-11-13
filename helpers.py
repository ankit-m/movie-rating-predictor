import numpy as np
import matplotlib.pyplot as plt

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

def calc_accuracy (h, y):
    count = 0
    for i in range(len(h)):
        if abs(h[i] - y[i]) <= 0.5:
            count += 1
    return count/float(len(h))

def plot_predictions (h, y):
    plt.plot(h, color='r', linestyle=':', marker='o')
    plt.plot(y, color='b', linestyle=':', marker='o')
    plt.show()
