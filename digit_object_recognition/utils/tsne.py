"""
    Import it to where you need to plot T-SNE
"""


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE


# Pre-defined color
color_list = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5DC',
              '#FFE4C4', '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A',
              '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED',
              '#FFF8DC', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B',
              '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00',
              '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F',
              '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF',
              '#B22222', '#228B22', '#FF00FF', '#DCDCDC', '#FFD700', '#DAA520',
              '#BEBEBE', '#808080', '#00FF00', '#008000', '#ADFF2F', '#FF69B4',
              '#CD5C5C', '#4B0082', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00',
              '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#D3D3D3']
user_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def plot_embedding(X, title=None, filename='0', batch_size=40):
    """
    Plot T-SNE embedding
    :param X: (batch_size x user_num) x 2
    :param title:
    :param filename:
    :param batch_size:
    :return:
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], user_list[i // batch_size],
                 color=color_list[i // batch_size])

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            shown_images = np.r_[shown_images, [X[i]]]
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    # save figure
    plt.savefig(filename + '.pdf')


if __name__ == '__main__':

    tsne = TSNE(n_components=2, init='pca', method='exact')

    user_num = 10
    batch_size = 40
    vector_len = 100

    # make sure the input is numpy array
    # input can be activation
    array = np.random.normal(0, 1, [user_num * batch_size, vector_len])
    input_vec = np.array(array)
    x_tsne = tsne.fit_transform(input_vec)
    plot_embedding(x_tsne, "t-SNE embedding of the activation")
