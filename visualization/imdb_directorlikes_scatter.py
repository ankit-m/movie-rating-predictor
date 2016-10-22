import matplotlib.pyplot as plt

def plot_data (df):
    df.plot(kind='scatter', x='imdb_score', y='director_facebook_likes', c='1' , edgecolors='0.35');
    plt.xlabel('IMDB Rating')
    plt.ylabel('Director Popularity (facebook likes)')
    plt.ylim([0, 25000])
    plt.show()
