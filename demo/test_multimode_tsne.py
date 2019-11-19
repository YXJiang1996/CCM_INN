import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

from itertools import combinations

column_names = ['Black', 'Yellow', 'Pink', 'Purple', 'Blue', 'Gray']


def generate(n_samples, n_type):
    np.random.seed(1)
    samples = np.random.uniform(0, 1, size=(n_samples, n_type))

    # choose three painting to generate recipe
    r = list(combinations(np.arange(0, n_type, 1), 3))
    r_num = r.__len__()
    n = n_samples // r_num
    for i in range(r_num - 1):
        samples[i * n:(i + 1) * n, r[i]] = 0.
    samples[(r_num - 1) * n:, r[r_num - 1]] = 0.

    # 对数据进行随机化
    shuffling = np.random.permutation(n_samples)
    samples = samples[shuffling]

    return samples


def plot(c_rev, c_test):
    tsne = TSNE(n_components=2, init='pca')
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    N_samp = 256

    cnt = 0
    for i in range(3):
        for j in range(3):
            rev_x = c_rev[cnt * 256:(cnt + 1) * 256, :]
            # 假设涂料浓度小于一定值，就不需要这种涂料
            rev_x = np.where(rev_x < 0.5, 0, rev_x)

            # feature scaling
            test_x = c_test[cnt, :].reshape(1, c_test[cnt, :].shape[-1])
            plot_x = np.concatenate((rev_x, test_x), axis=0)

            # use pca to decrease dimensionality
            x_norm = pd.DataFrame(
                plot_x,
                columns=column_names)

            '''
            class_norm = pd.DataFrame(
                np.concatenate((np.ones(N_samp).reshape(N_samp, 1), np.zeros(1).reshape(1, 1)), axis=0),
                columns=['class'])
            '''

            # 根据需要的涂料种类（需要为1，不需要为0）将配方分类
            classes = np.zeros(N_samp).reshape(N_samp, 1)
            paint_needed = np.where(rev_x == 0, 0, 1)
            true_recipe_type = 0
            true_needed = np.where(test_x == 0, 0, 1)
            for paint_no in range(6):
                classes[:, 0] += paint_needed[:, paint_no] * 2 ** paint_no
                true_recipe_type += true_needed[:, paint_no] * 2 ** paint_no
            class_norm = pd.DataFrame(np.concatenate((classes, np.zeros(1).reshape(1, 1)), axis=0),
                                      columns=['class'])

            '''
            complete_data = pd.concat([x_norm, class_norm], axis=1)
            class_data = complete_data['class']

            axes[i, j].clear()
            
            for recipe_type in np.array(class_norm[:-1].drop_duplicates()).reshape(1, -1).tolist()[0]:
                recipes = complete_data[class_data == recipe_type]
                recipe_class = pd.DataFrame((recipe_type * np.ones(recipes.shape[0])).reshape(recipes.shape[0], 1),
                                            columns=['class'])
                if recipe_type == true_recipe_type[0]:
                    recipes = pd.concat([recipes, complete_data[class_data == 0]], axis=0)
                    recipe_class = pd.DataFrame(np.concatenate(
                        ((recipe_type * np.ones(recipes.shape[0]-1)).reshape(recipes.shape[0]-1, 1),
                         np.zeros(1).reshape(1, 1)), axis=0), columns=['class'])

                data_plot = pd.concat([pd.DataFrame(tsne.fit_transform(recipes[column_names])), recipe_class],
                                      axis=1)
                new_class_data = data_plot['class']

                axes[i, j].scatter(data_plot[new_class_data == recipe_type][0],
                                   data_plot[new_class_data == recipe_type][1],
                                   s=2, alpha=0.5)
                if recipe_type == true_recipe_type[0]:
                    axes[i, j].scatter(data_plot[new_class_data == 0][0], data_plot[new_class_data == 0][1],
                                       marker='+', s=10)
            '''

            data_plot = pd.concat([pd.DataFrame(tsne.fit_transform(x_norm)), class_norm], axis=1)

            class_data = data_plot['class']

            # plot the predicted and the true recipe
            axes[i, j].clear()
            recipe_classes = np.array(class_norm[:-1].drop_duplicates()).reshape(1, -1).tolist()[0]
            for recipe_class in recipe_classes:
                axes[i, j].scatter(data_plot[class_data == recipe_class][0],
                                   data_plot[class_data == recipe_class][1],
                                   s=2, alpha=0.5)
            axes[i, j].scatter(data_plot[class_data == 0][0], data_plot[class_data == 0][1], marker='+', s=10)

            cnt += 1

    # plt.legend()
    plt.show()


def main():
    # 2^10 样本，每个样本6个元素
    np.random.seed(1)
    samples = generate(2 ** 10, 6)
    x = samples[-9:, :]
    rev_x = np.random.uniform(0, 1, size=(256 * 9, 6))
    for i in range(9):
        rev_x[i * 256:(i + 1) * 256, :] += x[i, :]
    plot(rev_x, x)


main()
