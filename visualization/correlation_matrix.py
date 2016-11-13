import matplotlib.pyplot as plt

def plot_data (data, ids):
    correlation_matrix = data[ids].corr()
    plt.imshow(correlation_matrix, cmap=plt.cm.Spectral_r, interpolation='nearest')
    plt.xticks(range(len(correlation_matrix)), ids, fontsize=10, rotation='vertical')
    plt.yticks(range(len(correlation_matrix)), ids, fontsize=10)
    plt.colorbar()
    plt.show()
