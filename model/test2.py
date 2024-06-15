import torch
import torch.nn as nn
import math
import numpy as np

# 实际距离（边）
x = torch.randn(2, 3, 120, 25)
x.reshape(2, 120, 3, 25)

def geometric_distance(x, i, j):
    summ = torch.zeros(x.shape[0], x.shape[1], x.shape[-1])
    for xyz in range(3):
        summ = summ + (x[:][:][xyz][i] - x[:][:][xyz][j]) ** 2
    return torch.sqrt(summ)

t = torch.tensor([[[[0,1,2,-1,-2],[0,1,2,-1,-2],[0,1,2,-1,-2]]]])
print(t.shape)
print(geometric_distance(torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), 0, 1).shape)

E = torch.zeros((25, 25))
D = torch.zeros((25, 25)) - 1
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
for (i, j) in inward:
    E[i][j] = 1
    E[j][i] = 1
print(E)
for i in range(25):
    D[i][i] = 0
    for j in range(25):
        if E[i][j] == 1:
            D[i][j] = geometric_distance(x, i, j)
print(D)


def floyd(dist):
    # 初始化
    V = dist.shape[-1]
    d = dist.clone()
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if d[i, k] != -1 and d[k, j] != -1:  # 跳过不连通的情况
                    if d[i, j] == -1 or d[i, j] > d[i, k] + d[k, j]:
                        d[i, j] = d[i, k] + d[k, j]
    return d


DD = floyd(D)
print(DD)
# x = torch.tensor([[0,1,2],[0,1,2],[0,1,2]])
# print(geometric_distance(x,0,2))
