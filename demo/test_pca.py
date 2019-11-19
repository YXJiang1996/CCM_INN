import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import TSNE

from itertools import combinations

column_names = ['Black', 'Yellow', 'Pink', 'Purple', 'Blue', 'Gray']


def generate(n_samples, n_type):
    np.random.seed(1)
    samples = np.random.uniform(0, 1, size=(n_samples, n_type))
    classes = np.zeros(n_samples).reshape(n_samples, 1)

    # choose three painting to generate recipe
    r = list(combinations(np.arange(0, n_type, 1), 3))
    r_num = r.__len__()
    n = n_samples // r_num
    for i in range(r_num - 1):
        samples[i * n:(i + 1) * n, r[i]] = 0.
        classes[i * n:(i + 1) * n, 0] = i + 1
    samples[(r_num - 1) * n:, r[r_num - 1]] = 0.
    classes[(r_num - 1) * n:, 0] = r_num

    # 对数据进行随机化
    shuffling = np.random.permutation(n_samples)
    samples = pd.DataFrame(samples[shuffling], columns=column_names)
    classes = pd.DataFrame(classes[shuffling], columns=['Class'])
    samples = pd.concat([samples, classes], axis=1)
    return samples, r_num


def plot(X, r_num):
    classes = X['Class']
    x = X.ix[:, column_names[0]:column_names[-1]]
    x_norm = (x - x.min()) / (x.max() - x.min())  # feature scaling

    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    tsne = TSNE(n_components=2, init='pca')
    transformed = pd.DataFrame(tsne.fit_transform(x_norm))

    for i in range(r_num):
        plt.scatter(transformed[classes == i + 1][0], transformed[classes == i + 1][1], label='Class%d' % (i + 1))
    # plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Class 2', c='blue')
    # plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Class 3', c='lightgreen')
    plt.scatter(transformed[classes == r_num + 1][0], transformed[classes == r_num + 1][1],
                label='Class%d' % (r_num + 1))

    # plt.legend()
    plt.show()


def main():
    # 2^10 样本，每个样本6个元素
    samples, r_num = generate(2 ** 10, 6)
    plot(samples, r_num)


main()
