import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

def scale_data (x_train, x_test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return (x_train, x_test)

def plot_predictions (h, y):
    plt.plot(h, color='r', linestyle=':', marker='o')
    plt.plot(y, color='b', linestyle=':', marker='o')
    plt.show()
