import matplotlib.pyplot as plt

def plot_data (df):
    df1 = df[['imdb_score', 'country']]
    color = dict(boxes='DarkGray', whiskers='DarkRed', medians='DarkBlue', caps='Gray')
    df1.boxplot(by='country', rot=90)
    plt.show()
