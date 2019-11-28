import numpy as np
import torch
import torch.utils.data

ingredients = np.array([
    [0.3427, 0.4346, 0.4409, 0.4418, 0.4449, 0.4509, 0.4701, 0.5372, 0.6944, 0.8247, 0.8914, 0.9259, 0.9373, 0.9418,
     0.9441, 0.9462, 0.9474, 0.948, 0.9483, 0.948, 0.9484, 0.9479, 0.9475, 0.9472, 0.9468, 0.9472, 0.9465, 0.9442,
     0.9423, 0.9387, 0.933],
    [0.2073, 0.2314, 0.2324, 0.2326, 0.2316, 0.2308, 0.23, 0.2295, 0.2283, 0.2268, 0.2266, 0.2254, 0.2244, 0.2231,
     0.2221, 0.2208, 0.2202, 0.2193, 0.2183, 0.2171, 0.2163, 0.2147, 0.2133, 0.2119, 0.2107, 0.2094, 0.2086, 0.2077,
     0.2035, 0.2023, 0.2005]])
initial_concentration = np.array([0.51, 0.51])
background = np.array([0.4519, 0.7445, 0.8898, 0.9311, 0.9374, 0.9396, 0.9426, 0.9435, 0.9453, 0.9456, 0.9472, 0.9475,
                       0.9473, 0.9477, 0.9474, 0.9472, 0.9469, 0.9464, 0.9459, 0.945, 0.9446, 0.9434, 0.9428, 0.9422,
                       0.9418, 0.9421, 0.9416, 0.9396, 0.9375, 0.934, 0.9281])
# CIE标准照明体D65光源，10°视场
optical_relevant = np.array([[0.136, 0.667, 1.644, 2.348, 3.463, 3.733, 3.065, 1.934, 0.803, 0.151, 0.036, 0.348, 1.062,
                              2.192, 3.385, 4.744, 6.069, 7.285, 8.361, 8.537, 8.707, 7.946, 6.463, 4.641, 3.109, 1.848,
                              1.053, 0.575, 0.275, 0.120, 0.059],
                             [0.014, 0.069, 0.172, 0.289, 0.560, 0.901, 1.300, 1.831, 2.530, 3.176, 4.337, 5.629, 6.870,
                              8.112, 8.644, 8.881, 8.583, 7.922, 7.163, 5.934, 5.100, 4.071, 3.004, 2.031, 1.295, 0.741,
                              0.416, 0.225, 0.107, 0.046, 0.023],
                             [0.613, 3.066, 7.820, 11.589, 17.755, 20.088, 17.697, 13.025, 7.703, 3.889, 2.056, 1.040,
                              0.548, 0.282, 0.123, 0.036, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                              0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])
perfect_white = np.array([[94.83], [100.00], [107.38]])


def generate(tot_dataset_size, model='km', ydim=31, sigma=0.1, prior_bound=[0, 1, 0, 1], seed=0):
    np.random.seed(seed)
    N = tot_dataset_size

    concentrations = np.random.uniform(0, 1, size=(N, 2))
    concentrations[:, 0] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, 0]
    concentrations[:, 1] = prior_bound[2] + (prior_bound[3] - prior_bound[2]) * concentrations[:, 1]

    xvec = np.arange(400, 710, (700 - 400) / (ydim - 1))
    xidx = np.arange(0, ydim, 1)
    init_conc_array = initial_concentration.repeat(ingredients.shape[1]).reshape(2, -1)

    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.array(
            [concentrations[:, 0] * fst[0, i] + concentrations[:, 1] * fst[1, i] + np.ones(N) * fsb[i]
             for i in xidx])

        reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        reflectance = reflectance.transpose()

    elif model == 'four_flux':
        ones_background = np.ones_like(background)
        fsb = (8 * background + (ones_background - 6 * background) *
               ((4 * background ** 2 - 4 * background + 25 * ones_background) ** 0.5)
               + 12 * background ** 2 + 5 * ones_background) / (48 * background)

        ones_ingredients = np.ones_like(ingredients).reshape(ingredients.shape[0], ingredients.shape[1])
        fst = ((8 * ingredients + (ones_ingredients - 6 * ingredients) *
                ((4 * ingredients ** 2 - 4 * ingredients + 25 * ones_ingredients) ** 0.5)
                + 12 * ingredients ** 2 + 5 * ones_ingredients) / (48 * ingredients) - fsb) / init_conc_array

        fss = np.array(
            [concentrations[:, 0] * fst[0, i] + concentrations[:, 1] * fst[1, i] + np.ones(N) * fsb[i]
             for i in xidx])

        ones_fss = np.ones_like(fss).reshape(fss.shape[0], fss.shape[1])
        reflectance = 0.5 * (1 / ((4 * (fss ** 2) + 4 * fss) ** 0.5 + 2 * fss + ones_fss)) + 0.5 * (
                (((fss + ones_fss) * ((4 * (fss ** 2) + 4 * fss) ** 0.5)) + 2 * (fss ** 2) - 2 * ones_fss) / (
                2 * (fss + ones_fss) * (3 * fss - ones_fss) * (
                ((4 * (fss ** 2) + 4 * fss) ** 0.5) + 2 * fss + ones_fss)))
        reflectance = reflectance.transpose()

    else:
        print('Sorry no model of that name')
        exit(1)

    # randomise the data
    shuffling = np.random.permutation(N)
    concentrations = torch.tensor(concentrations[shuffling], dtype=torch.float)
    reflectance = torch.tensor(reflectance[shuffling], dtype=torch.float)

    return concentrations, reflectance, xvec


def get_lik(ydata, n_grid=64, model='km', sigma=None, xvec=None, bound=[0, 1, 0, 1]):
    mcx = np.linspace(bound[0], bound[1], n_grid)
    mcy = np.linspace(bound[2], bound[3], n_grid)
    dmcx = mcx[1] - mcx[0]
    dmcy = mcy[1] - mcy[0]

    init_conc_array = initial_concentration.repeat(ingredients.shape[1]).reshape(2, -1)

    diff = np.zeros((n_grid, n_grid))
    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array

        for i, c in enumerate(mcy):
            fss = np.array(
                [A * fst[0] + c * fst[1] + fsb for A in mcx])
            diff[i, :] = np.array([color_diff(ydata, p - ((p + 1) ** 2 - 1) ** 0.5 + 1) for p in fss])

    elif model == 'four_flux':
        ones_background = np.ones_like(background)
        fsb = (8 * background + (ones_background - 6 * background) *
               ((4 * background ** 2 - 4 * background + 25 * ones_background) ** 0.5)
               + 12 * background ** 2 + 5 * ones_background) / (48 * background)

        ones_ingredients = np.ones_like(ingredients).reshape(ingredients.shape[0], ingredients.shape[1])
        fst = ((8 * ingredients + (ones_ingredients - 6 * ingredients) *
                ((4 * ingredients ** 2 - 4 * ingredients + 25 * ones_ingredients) ** 0.5)
                + 12 * ingredients ** 2 + 5 * ones_ingredients) / (48 * ingredients) - fsb) / init_conc_array

        for i, c in enumerate(mcy):
            fss = np.array(
                [A * fst[0] + c * fst[1] + fsb for A in mcx])
            diff[i, :] = np.array([color_diff(ydata, 0.5 * (1 / ((4 * (p ** 2) + 4 * p) ** 0.5 + 2 * p + 1)) + 0.5 * (
                    (((p + 1) * ((4 * (p ** 2) + 4 * p) ** 0.5)) + 2 * (p ** 2) - 2 * 1) / (
                    2 * (p + 1) * (3 * p - 1) * (((4 * (p ** 2) + 4 * p) ** 0.5) + 2 * p + 1)))) for p in fss])

    else:
        print('Sorry no model of that name')
        exit(1)

    # normalise the posterior
    diff /= (np.sum(diff.flatten()) * dmcx * dmcy)

    # compute integrated probability outwards from max point
    diff = diff.flatten()
    idx = np.argsort(diff)[::-1]
    prob = np.zeros(n_grid * n_grid)
    prob[idx] = np.cumsum(diff[idx]) * dmcx * dmcy
    prob = prob.reshape(n_grid, n_grid)
    return mcx, mcy, prob


def color_diff(reflectance1, reflectance2):
    tri1 = np.dot(optical_relevant, reflectance1.reshape(31, 1))
    tri2 = np.dot(optical_relevant, reflectance2.reshape(31, 1))

    lab1 = xyz2lab(tri1)
    lab2 = xyz2lab(tri2)
    delta_lab = lab1 - lab2

    diff = (delta_lab[0] ** 2 + delta_lab[1] ** 2 + delta_lab[2] ** 2) ** (1 / 2)
    return diff


def xyz2lab(xyz):
    r = 0.008856
    lab = np.zeros(3 * 1)

    if xyz[0] / perfect_white[0] > r and xyz[1] / perfect_white[1] > r and xyz[2] / perfect_white[2] > r:
        lab[0] = (xyz[1] / perfect_white[1]) ** (1 / 3) * 116 - 16
        lab[1] = ((xyz[0] / perfect_white[0]) ** (1 / 3) - (xyz[1] / perfect_white[1]) ** (1 / 3)) * 500
        lab[2] = ((xyz[1] / perfect_white[1]) ** (1 / 3) - (xyz[2] / perfect_white[2]) ** (1 / 3)) * 200
    else:
        lab[0] = (xyz[1] / perfect_white[1]) * 903.3
        lab[1] = (xyz[0] / perfect_white[0] - xyz[1] / perfect_white[1]) * 3893.5
        lab[2] = (xyz[1] / perfect_white[1] - xyz[2] / perfect_white[2]) * 1557.4

    return lab


def color_diff_tensor(reflectance1, reflectance2):
    optical_relevant_torch = torch.from_numpy(optical_relevant).float()
    tri1 = optical_relevant_torch.mm(torch.reshape(reflectance1, (reflectance1.size(-1), 1)))
    tri2 = optical_relevant_torch.mm(torch.reshape(reflectance2, (reflectance2.size(-1), 1)))

    lab1 = xyz2lab_tensor(tri1)
    lab2 = xyz2lab_tensor(tri2)
    delta_lab = lab1 - lab2

    diff = (delta_lab[0] ** 2 + delta_lab[1] ** 2 + delta_lab[2] ** 2) ** (1 / 2)
    return diff


def xyz2lab_tensor(xyz):
    r = 0.008856
    lab = torch.zeros([3, 1], dtype=torch.float)
    perfect_white_tensor = torch.from_numpy(perfect_white).float()

    if (xyz[0] / perfect_white_tensor[0] > r
            and xyz[1] / perfect_white_tensor[1] > r
            and xyz[2] / perfect_white_tensor[2] > r):
        lab[0] = (xyz[1] / perfect_white_tensor[1]) ** (1 / 3) * 116 - 16
        lab[1] = ((xyz[0] / perfect_white_tensor[0]) ** (1 / 3) - (xyz[1] / perfect_white_tensor[1]) ** (1 / 3)) * 500
        lab[2] = ((xyz[1] / perfect_white_tensor[1]) ** (1 / 3) - (xyz[2] / perfect_white_tensor[2]) ** (1 / 3)) * 200
    else:
        lab[0] = (xyz[1] / perfect_white_tensor[1]) * 903.3
        lab[1] = (xyz[0] / perfect_white_tensor[0] - xyz[1] / perfect_white_tensor[1]) * 3893.5
        lab[2] = (xyz[1] / perfect_white_tensor[1] - xyz[2] / perfect_white_tensor[2]) * 1557.4

    return lab


def recipe_reflectance(recipes, model='km'):
    xidx = np.arange(0, 31, 1)
    init_conc_array = np.repeat(initial_concentration.reshape(2, 1), 31).reshape(2, 31)
    reflectance = np.zeros(31 * recipes.shape[0]).reshape(31, recipes.shape[0])

    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.zeros(31 * recipes.shape[0]).reshape(31, recipes.shape[0])
        for i in xidx:
            for j in range(2):
                fss[i, :] += recipes[:, j] * fst[j, i]
            fss[i, :] += np.ones(recipes.shape[0]) * fsb[i]
        reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        reflectance = reflectance.transpose()

    elif model == 'four_flux':
        ones_background = np.ones_like(background)
        fsb = (8 * background + (ones_background - 6 * background) *
               ((4 * background ** 2 - 4 * background + 25 * ones_background) ** 0.5)
               + 12 * background ** 2 + 5 * ones_background) / (48 * background)

        ones_ingredients = np.ones_like(ingredients).reshape(ingredients.shape[0], ingredients.shape[1])
        fst = ((8 * ingredients + (ones_ingredients - 6 * ingredients) *
                ((4 * ingredients ** 2 - 4 * ingredients + 25 * ones_ingredients) ** 0.5)
                + 12 * ingredients ** 2 + 5 * ones_ingredients) / (48 * ingredients) - fsb) / init_conc_array

        fss = np.zeros(31 * recipes.shape[0]).reshape(31, recipes.shape[0])
        for i in xidx:
            for j in range(2):
                fss[i, :] += recipes[:, j] * fst[j, i]
            fss[i, :] += np.ones(recipes.shape[0]) * fsb[i]

        ones_fss = np.ones_like(fss).reshape(fss.shape[0], fss.shape[1])
        reflectance = 0.5 * (1 / ((4 * (fss ** 2) + 4 * fss) ** 0.5 + 2 * fss + ones_fss)) + 0.5 * (
                (((fss + ones_fss) * ((4 * (fss ** 2) + 4 * fss) ** 0.5)) + 2 * (fss ** 2) - 2 * ones_fss) / (
                2 * (fss + ones_fss) * (3 * fss - ones_fss) * (
                ((4 * (fss ** 2) + 4 * fss) ** 0.5) + 2 * fss + ones_fss)))
        reflectance = reflectance.transpose()

    else:
        print('Sorry no model of that name')
        exit(1)

    return reflectance


def four_flux():
    x = 0.57269
    y = (1 / 2) * (1 / ((4 * (x ** 2) + 4 * x) ** (1 / 2) + 2 * x + 1)) + (1 / 2) * (
            ((x + 1) * ((4 * (x ** 2) + 4 * x) ** (1 / 2)) + 2 * (x ** 2) - 2) / (
            2 * (x + 1) * (3 * x - 1) * ((4 * (x ** 2) + 4 * x) ** (1 / 2) + 2 * x + 1)))
    x_rev = (8 * y - 6 * y * ((4 * (y ** 2) - 4 * y + 25) ** (1 / 2)) + (4 * (y ** 2) - 4 * y + 25) ** (
            1 / 2) + 12 * (y ** 2) + 5) / (48 * y)
    print(y)
    print(x_rev)
