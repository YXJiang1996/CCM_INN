import numpy as np
import torch
import torch.utils.data

# 黑、黄、粉、紫、蓝、灰白
ingredients0 = np.array([
    [0.2072717, 0.2314336, 0.2323573, 0.2326192, 0.2315759, 0.2308346, 0.2299640, 0.2295049, 0.2283248, 0.2267959,
     0.2265563, 0.2253986, 0.2243909, 0.2231159, 0.2221052, 0.2208150, 0.2201522, 0.2193270, 0.2182626, 0.2170505,
     0.2162608, 0.2146650, 0.2133304, 0.2119302, 0.2106560, 0.2094081, 0.2085703, 0.2077338, 0.2034682, 0.2023278,
     0.2005492],
    [0.3426769, 0.4345741, 0.4408946, 0.4417734, 0.4448756, 0.4509351, 0.4701447, 0.5371643, 0.6943693, 0.8247025,
     0.8914063, 0.9259378, 0.9373208, 0.9417730, 0.9441082, 0.9462348, 0.9473710, 0.9479665, 0.9482779, 0.9480201,
     0.9483707, 0.9478786, 0.9474954, 0.9472089, 0.9468048, 0.9471580, 0.9465347, 0.9441642, 0.9423221, 0.9387001,
     0.9329507],
    [0.2708842, 0.2955388, 0.2835928, 0.2703437, 0.2596675, 0.2515463, 0.2464779, 0.2449044, 0.2445325, 0.2466007,
     0.2538134, 0.2617629, 0.2677513, 0.2809876, 0.3157711, 0.3914984, 0.5052982, 0.6254060, 0.7242598, 0.8023202,
     0.8626911, 0.8989766, 0.9154678, 0.9242462, 0.9278284, 0.9308498, 0.9321416, 0.9321451, 0.9319565, 0.9283142,
     0.9233366],
    [0.3734002, 0.5346663, 0.5803110, 0.5968833, 0.5961596, 0.5762038, 0.5414493, 0.5004809, 0.4368144, 0.3691231,
     0.3134314, 0.2668127, 0.2305206, 0.2116322, 0.2072988, 0.2046728, 0.2012968, 0.2056570, 0.2243125, 0.2557296,
     0.2817601, 0.2780255, 0.2626838, 0.2666584, 0.3057532, 0.3696144, 0.4543115, 0.5407568, 0.6394141, 0.6732088,
     0.7227149],
    [0.3747776, 0.5648752, 0.6199736, 0.6661413, 0.7298144, 0.7726266, 0.7812571, 0.7792725, 0.7607376, 0.7320930,
     0.6943575, 0.6432314, 0.5800607, 0.5089791, 0.4396545, 0.3704733, 0.3041328, 0.2567624, 0.2272359, 0.2094773,
     0.1948587, 0.1825973, 0.1767187, 0.1732840, 0.1733995, 0.1762372, 0.1840847, 0.1860129, 0.1805977, 0.1722357,
     0.1589434],
    [0.3737892, 0.5056624, 0.5303422, 0.5383733, 0.5443333, 0.5562689, 0.5746784, 0.5999998, 0.6406719, 0.6941598,
     0.7595113, 0.8284632, 0.8829190, 0.9176573, 0.9354603, 0.9417856, 0.9447337, 0.9462350, 0.9468061, 0.9464547,
     0.9465482, 0.9460330, 0.9458108, 0.9456140, 0.9452482, 0.9455251, 0.9448371, 0.9429901, 0.9404516, 0.9368277,
     0.9308580]])
initial_concentration0 = np.array([0.51, 0.51, 0.6, 0.498, 0.508, 0.54])
background0 = np.array(
    [0.4519222, 0.7445221, 0.8898484, 0.9311465, 0.9374331, 0.9395607, 0.9426168, 0.9435278, 0.9453126, 0.9456188,
     0.9471663, 0.9475049, 0.9473154, 0.9476760, 0.9473940, 0.9471831, 0.9469163, 0.9463666, 0.9459230, 0.9450417,
     0.9446035, 0.9434228, 0.9427990, 0.9421890, 0.9418073, 0.9421464, 0.9415964, 0.9395502, 0.9375165, 0.9339727,
     0.9281333])
info = background0.copy()
for i, c in enumerate(initial_concentration0):
    info = np.append(info, c)
    info = np.append(info, ingredients0[i])

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


def generate(tot_dataset_size, model='km', ydim=31, info=info, prior_bound=[0, 1, 0, 1], seed=0):
    np.random.seed(seed)
    N = tot_dataset_size

    # Get painting information
    background = info[:ydim]
    colors = np.arange(0, (info.shape[-1] - ydim) // (ydim + 1), 1)
    initial_concentration = np.zeros(colors.size * 1)
    ingredients = np.zeros(colors.size * ydim).reshape(colors.size, ydim)
    for i, c in enumerate(colors):
        initial_concentration[i] = info[ydim + i * (ydim + 1)]
        ingredients[i] = info[ydim + i * (ydim + 1) + 1:ydim + (i + 1) * (ydim + 1)]

    if model == 'km':
        concentrations = np.random.uniform(0, 1, size=(N, colors.size))
        for i in colors:
            concentrations[:, i] = prior_bound[0] + (prior_bound[1] - prior_bound[0]) * concentrations[:, i]

        xvec = np.arange(400, 710, (700 - 400) / (ydim - 1))
        xidx = np.arange(0, ydim, 1)
        init_conc_array = np.repeat(initial_concentration.reshape(colors.size, 1), ydim).reshape(colors.size, ydim)

        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.zeros(N * ydim).reshape(ydim, N)
        for i in xidx:
            for j in colors:
                fss[i, :] += concentrations[:, j] * fst[j, i]
            fss[i, :] += np.ones(N) * fsb[i]

        reflectance = fss - ((fss + 1) ** 2 - 1) ** 0.5 + 1
        reflectance = reflectance.transpose()

    elif model == 'four_flux':
        print('Sorry the model have not implemented yet')
        exit(1)

    else:
        print('Sorry no model of that name')
        exit(1)

    # randomise the data
    shuffling = np.random.permutation(N)
    concentrations = torch.tensor(concentrations[shuffling], dtype=torch.float)
    reflectance = torch.tensor(reflectance[shuffling], dtype=torch.float)

    return concentrations, reflectance, xvec, info


def get_lik(ydata, n_grid=64, info=info, model='km', bound=[0, 1, 0, 1]):
    mcx = np.linspace(bound[0], bound[1], n_grid)
    dmcx = mcx[1] - mcx[0]

    # Get painting information
    ydim = ydata.size
    background = info[:ydim]
    colors = np.arange(0, (info.shape[-1] - ydim) // (ydim + 1), 1)
    initial_concentration = np.zeros(colors.size * 1)
    ingredients = np.zeros(colors.size * ydim).reshape(colors.size, ydim)
    for i in colors:
        initial_concentration[i] = info[ydim + i * (ydim + 1)]
        ingredients[i] = info[ydim + i * (ydim + 1) + 1:ydim + (i + 1) * (ydim + 1)]

    init_conc_array = np.repeat(initial_concentration.reshape(colors.size, 1), ydim).reshape(colors.size, ydim)

    # concentrations of painting
    # 这里缺乏灵活性，之后再改
    cons = np.zeros((n_grid ** colors.size, colors.size))
    yidx = np.arange(0, ydim, 1)
    for i, c in enumerate(mcx):
        for j, d in enumerate(mcx):
            for k, e in enumerate(mcx):
                for l, f in enumerate(mcx):
                    for m, g in enumerate(mcx):
                        for n, h in enumerate(mcx):
                            cons[i * (n_grid ** 5) + j * (n_grid ** 4) + k * (n_grid ** 3) +
                                 l * (n_grid ** 2) + m * n_grid + n] = [c, d, e, f, g, h]

    diff = np.zeros(n_grid ** colors.size)
    if model == 'km':
        fsb = (np.ones_like(background) - background) ** 2 / (background * 2)
        fst = ((np.ones_like(ingredients) - ingredients) ** 2 / (ingredients * 2) - fsb) / init_conc_array
        fss = np.zeros((n_grid ** colors.size, yidx.size))
        for i in yidx:
            for j in colors:
                fss[:, i] += cons[:, j] * fst[j, i]
            fss[:, i] += np.ones(n_grid ** colors.size) * fsb[i]
        diff = np.array([color_diff(ydata, p - ((p + 1) ** 2 - 1) ** 0.5 + 1) for p in fss]).transpose()
        '''
        for i, c in enumerate(mcy):
            fss = np.array(
                [A * fst[0] + c * fst[1] + fsb for A in mcx])
            diff[i, :] = np.array([color_diff(ydata, p - ((p + 1) ** 2 - 1) ** 0.5 + 1) for p in fss])
        '''

    elif model == 'four_flux':
        print('Sorry the model have not implemented yet')
        exit(1)

    else:
        print('Sorry no model of that name')
        exit(1)

    # normalise the posterior
    diff /= (np.sum(diff.flatten()) * (dmcx ** colors.size))

    # compute integrated probability outwards from max point
    diff = diff.flatten()
    idx = np.argsort(diff)[::-1]
    prob = np.zeros(n_grid ** colors.size)
    prob[idx] = np.cumsum(diff[idx]) * (dmcx ** colors.size)
    return mcx, cons, prob


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

'''
get_lik(
    ydata=np.array(
        [0.2787464, 0.3808284, 0.4138503, 0.4381292, 0.4667822, 0.5057398, 0.5647132, 0.6312735, 0.7014090, 0.7491776,
         0.7615933, 0.7522775, 0.7260387, 0.6904571, 0.6457943, 0.5945013, 0.5372259, 0.4788299, 0.4185695, 0.3582843,
         0.3021949, 0.2646684, 0.2481450, 0.2407368, 0.2375998, 0.2367689, 0.2414787, 0.2570425, 0.2800411, 0.2906137,
         0.3009636]), n_grid=4)
'''
