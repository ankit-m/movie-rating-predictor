import matplotlib.pyplot as plt

def plot_data(df):
    df.plot(kind='scatter', x='cast_total_facebook_likes', y='imdb_score', c='0.35' , edgecolors='w');
    plt.xlabel('Cast Popularity (number of facebook likes)')
    plt.ylabel('IMDB Rating')
    plt.xlim([0, 1e5])
    plt.show()
