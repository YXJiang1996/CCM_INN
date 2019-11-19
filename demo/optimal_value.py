import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim
import numpy as np


def Fun(x, y):  # 原函数
    return x - y + 2 * x * x + 2 * x * y + y * y


# 初始化
fig = plt.figure()  # figure对象
ax = Axes3D(fig)  # Axes3D对象
X, Y = np.mgrid[-2:2:40j, -2:2:40j]  # 取样并作满射联合
Z = Fun(X, Y)  # 取样点Z坐标打表
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# 梯度下降
step = 0.0008  # 下降系数

x = torch.tensor(0., requires_grad=True)
y = torch.tensor(0., requires_grad=True)  # 初始选取一个点

optimizer = optim.SGD([x, y], lr=step)  # 更新规则

z = Fun(x, y)
tag_x = [x]
tag_y = [y]
tag_z = [z]  # 三个坐标分别打入表中，该表用于绘制点
Over = False
while not Over:
    optimizer.zero_grad()  # 清空x、y的梯度
    z.backward()  # 计算新梯度
    optimizer.step()  # 更新x、y的值

    if z - Fun(x, y) < 10e-9:  # 精度
        Over = True
    z = Fun(x, y)
    tag_x.append(x)
    tag_y.append(y)
    tag_z.append(z)  # 新点三个坐标打入表中

# 绘制点/输出坐标
ax.plot(tag_x, tag_y, tag_z, 'r.')
plt.title('(x,y)~(' + str(x.data) + "," + str(y.data) + ')')
plt.show()
