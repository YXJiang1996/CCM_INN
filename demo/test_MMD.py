import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


def frange(start, stop, step=1):
    i = start
    while i < stop:
        yield i
        i += step


def MMD_multiscale(x, y, a):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))

    # for a in [0.2, 0.5, 0.9, 1.3]:
    XX += a ** 2 * (a ** 2 + dxx) ** -1
    YY += a ** 2 * (a ** 2 + dyy) ** -1
    XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2. * XY)


def find_new_mmd_idx(a):
    aRev = a[::-1]
    for i, v in enumerate(a[-2::-1]):
        if v < aRev[i]:
            return min(len(a) - i, len(a) - 1)


def main():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(211)

    x = torch.randn(20, 6)
    y = torch.randn(20, 6)
    losses = []
    alpha = []
    for a in np.logspace(np.log10(0.5), np.log10(500), num=2000):
        alpha.append(a)
        losses.append(MMD_multiscale(x, y, a))

    find_new_mmd_idx(losses)
    ax.plot(alpha, losses, 'b')
    ax.set_xlabel(r'alpha')
    ax.set_ylabel(r'MMD')
    plt.savefig('MMD.png')
    plt.close()


main()
