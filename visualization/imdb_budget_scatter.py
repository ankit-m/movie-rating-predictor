import matplotlib.pyplot as plt

def plot_data (df):
    df.plot(kind='scatter', x='budget', y='imdb_score', c='0.35', s=df['gross']*5e-7, edgecolors='w');
    plt.xlabel('Budget (in hundred million dollars)')
    plt.ylabel('IMDB Rating')
    plt.xlim([0, 0.3e9])
    plt.show()
