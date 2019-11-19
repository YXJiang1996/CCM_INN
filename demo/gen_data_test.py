import numpy as np
from itertools import combinations

N = 1024
data = np.random.uniform(0, 1, size=(N, 6))

r = list(combinations(np.arange(0, 6, 1), 3))

r_num = r.__len__()
n = N // r_num
for i in range(r_num - 1):
    data[i * n:(i + 1) * n, r[i]] = 0.
data[(r_num - 1) * n:, r[r_num - 1]] = 0.

print("exit")
