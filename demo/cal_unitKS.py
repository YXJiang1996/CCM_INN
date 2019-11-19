import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def plot_unit_ks(c, unit_ks, img_name="unit_ks.png"):
    for i in np.arange(0, unit_ks.shape[-1], 1):
        plt.plot(c, unit_ks[:, i])
    plt.xlabel('c')
    plt.ylabel('unit K/S')
    plt.savefig(img_name)
    plt.close()


def cal_unit_ks_before(ks, blank_ks, c):
    return (ks - blank_ks) / c


def cal_unit_ks_after(ks, blank_ks, c, a, b):
    return (ks - blank_ks - a * (c ** 2) - b * (c ** 3)) / c


def loss(a1, a2, paintKS, blankKS, c):  # 原函数
    val = 0.0
    for i in np.arange(0, c.shape[-1], 1):
        for j in np.arange(i + 1, c.shape[-1], 1):
            val += torch.sum((((paintKS[i, :] - blankKS) / c[i] - a1 * c[i] - a2 * (c[i] ** 2)) - (
                    (paintKS[j, :] - blankKS) / c[j] - a1 * c[j] - a2 * (c[j] ** 2))) ** 2)
    return val


def adjust_para(paintKS, blankKS, c):
    a1 = torch.tensor(0., requires_grad=True)
    a2 = torch.tensor(0., requires_grad=True)  # 初始选取一个点

    optimizer = optim.Adam([a1, a2], lr=0.001)  # 更新规则

    losses = loss(a1, a2, paintKS, blankKS, c)
    over = False
    while not over:
        # print(losses)
        optimizer.zero_grad()  # 清空x、y的梯度
        losses.backward()  # 计算新梯度
        optimizer.step()  # 更新x、y的值

        if losses - loss(a1, a2, paintKS, blankKS, c) < 10e-9:  # 精度
            over = True
        losses = loss(a1, a2, paintKS, blankKS, c)

    return a1, a2


def main():
    paintRho = np.array([[0.3825898, 0.5427000, 0.5903007, 0.6153954, 0.6398891, 0.6740621, 0.7217913, 0.7726238,
                          0.8240511, 0.8569441, 0.8651740, 0.8579772, 0.8394680, 0.8127729, 0.7796661, 0.7386160,
                          0.6909225, 0.6389342, 0.5800068, 0.5178735, 0.4553084, 0.4110886, 0.3895794, 0.3785585,
                          0.3733844, 0.3700970, 0.3768114, 0.3948591, 0.4217977, 0.4312306, 0.4420620],
                         [0.3075271, 0.3970215, 0.4258692, 0.4498420, 0.4777384, 0.5187902, 0.5789000, 0.6485421,
                          0.7203640, 0.7698454, 0.7815039, 0.7697773, 0.7418166, 0.7037384, 0.6568713, 0.6019219,
                          0.5399272, 0.4776893, 0.4137468, 0.3495456, 0.2899768, 0.2505462, 0.2324735, 0.2232275,
                          0.2189511, 0.2174184, 0.2212247, 0.2353665, 0.2580868, 0.2678188, 0.2783012],
                         [0.2114880, 0.2507455, 0.2688738, 0.2892211, 0.3144960, 0.3527842, 0.4145680, 0.4928164,
                          0.5812670, 0.6469516, 0.6628808, 0.6468125, 0.6088670, 0.5602628, 0.5028209, 0.4396425,
                          0.3742497, 0.3143889, 0.2570285, 0.2052282, 0.1620530, 0.1352758, 0.1240267, 0.1177072,
                          0.1154153, 0.1142805, 0.1163994, 0.1241459, 0.1385358, 0.1454228, 0.1520851]])
    blankRho = np.array(
        [0.4519222, 0.7445221, 0.8898484, 0.9311465, 0.9374331, 0.9395607, 0.9426168, 0.9435278, 0.9453126, 0.9456188,
         0.9471663, 0.9475049, 0.9473154, 0.9476760, 0.9473940, 0.9471831, 0.9469163, 0.9463666, 0.9459230, 0.9450417,
         0.9446035, 0.9434228, 0.9427990, 0.9421890, 0.9418073, 0.9421464, 0.9415964, 0.9395502, 0.9375165, 0.9339727,
         0.9281333])
    c = np.array([0.159, 0.497, 1.509])
    paintKS = (np.ones_like(paintRho) - paintRho) ** 2 / (paintRho * 2)
    blankKS = (np.ones_like(blankRho) - blankRho) ** 2 / (blankRho * 2)

    unit_ks = []
    for i in np.arange(0, c.shape[-1], 1):
        unit_ks.append(cal_unit_ks_before(paintKS[i, :], blankKS, c[i]))
    plot_unit_ks(c, np.array(unit_ks), "unit_ks_before.png")

    paintRho *= (1-0.01)
    paintKS = (np.ones_like(paintRho) - paintRho) ** 2 / (paintRho * 2)

    paintKS = torch.from_numpy(paintKS).float()
    blankKS = torch.from_numpy(blankKS).float()
    c = torch.from_numpy(c).float()

    a1, a2 = adjust_para(paintKS, blankKS, c)
    print("a1=%.2f" % a1)
    print("a2=%.2f" % a2)

    unit_ks = []
    for i in np.arange(0, c.size(-1), 1):
        unit_ks.append(cal_unit_ks_after(paintKS[i, :], blankKS, c[i], a1, a2).detach().numpy())

    plot_unit_ks(c.detach().numpy(), np.array(unit_ks), "unit_ks_after.png")


main()
