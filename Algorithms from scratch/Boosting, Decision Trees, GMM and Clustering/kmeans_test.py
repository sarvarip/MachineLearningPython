import numpy as np
from data_loader import toy_dataset, load_digits
from kmeans import KMeans, KMeansClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import Figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    dmat = np.zeros((image.shape[0], image.shape[1], code_vectors.shape[0]))
    new_im = image

    for i in np.arange(image.shape[0]):
        for j in np.arange(image.shape[1]):
            for k in np.arange(code_vectors.shape[0]):
                dmat[i,j,k] =  np.sqrt(np.sum(np.square(np.subtract(code_vectors[k,:], image[i,j,:]))))
            idx = np.argmin(dmat[i,j,:])
            new_im[i,j,:] = code_vectors[idx,:]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

################################################################################
# KMeans on 2D toy dataset
# The dataset is generated from N gaussian distributions equally spaced on N radius circle.
# Here, N=4
# KMeans on this dataset should be able to identify the 4 clusters almost clearly.
################################################################################


def kmeans_toy():
    x, y = toy_dataset(4)
    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=y)
    fig.savefig('plots/toy_dataset_real_labels.png')

    fig.ax.scatter(x[:, 0], x[:, 1])
    fig.savefig('plots/toy_dataset.png')
    n_cluster = 4
    k_means = KMeans(n_cluster=n_cluster, max_iter=100, e=1e-8)
    centroids, membership, i = k_means.fit(x)

    assert centroids.shape == (n_cluster, 2), \
        ('centroids for toy dataset should be numpy array of size {} X 2'
            .format(n_cluster))

    assert membership.shape == (50 * n_cluster,), \
        'membership for toy dataset should be a vector of size 200'

    assert type(i) == int and i > 0,  \
        'Number of updates for toy datasets should be integer and positive'

    print('[success] : kmeans clustering done on toy dataset')
    print('Toy dataset K means clustering converged in {} steps'.format(i))

    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=membership)
    fig.ax.scatter(centroids[:, 0], centroids[:, 1], c='red')
    fig.savefig('plots/toy_dataset_predicted_labels.png')

    np.savez('results/k_means_toy.npz',
             centroids=centroids, step=i, membership=membership, y=y)

################################################################################
# KMeans for image compression
# Here we use k-means for compressing an image
# We load an image 'baboon.tiff',  scale it to [0,1] and compress it.
# The problem can be rephrased as --- "each pixel is a 3-D data point (RGB) and we want to map each point to N points or N clusters.
################################################################################


def kmeans_image_compression():
    im = plt.imread('baboon.tiff')
    N, M = im.shape[:2]
    im = im / 255

    # convert to RGB array
    data = im.reshape(N * M, 3)

    k_means = KMeans(n_cluster=16, max_iter=100, e=1e-6)
    centroids, _, i = k_means.fit(data)

    print('RGB centroids computed in {} iteration'.format(i))
    new_im = transform_image(im, centroids)

    assert new_im.shape == im.shape, \
        'Shape of transformed image should be same as image'

    mse = np.sum((im - new_im)**2) / (N * M)
    print('Mean square error per pixel is {}'.format(mse))
    plt.imsave('plots/compressed_baboon.png', new_im)

    np.savez('results/k_means_compression.npz', im=im, centroids=centroids,
             step=i, new_image=new_im, pixel_error=mse)

if __name__ == '__main__':
    kmeans_toy()
    kmeans_image_compression()
