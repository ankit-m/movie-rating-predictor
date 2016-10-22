import matplotlib.pyplot as plt

def plot_data (df):
    df['imdb_score'].hist(color = '0.35', bins = 40, edgecolor='w')
    plt.xlabel('IMDB Rating')
    plt.ylabel('Number of Movies')
    plt.show()
