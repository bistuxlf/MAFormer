import torch
import torch.nn as nn

# 输入矩阵shape: (N, C, V)
# def compute_distances(x):
#     # 将输入转换为形状 (N, T, V, C)
#     x = x.permute(0, 2, 3, 1)
#     # 计算每个节点之间的差异
#     diff = x.unsqueeze(-2) - x.unsqueeze(-3)
#     # 计算每个节点之间的欧几里得距离
#     distances = torch.norm(diff, dim=-1)
#     return distances
#
# t = torch.randn([2,3,120,25])
# print(t.shape)
# t_d = compute_distances(t)
# print(t_d.shape)
# net = nn.Embedding(25, 64)
# print(net(t_d).shape)

t = torch.tensor([[[[0,1,2,-1,-2,3],[0,1,2,-1,-2,3],[0,1,2,-1,-2,3]],[[0.1,1.1,2.1,-1.1,-2.1,3.1],[0.1,1.1,.1,-1.1,-2.1,3.1],[0,1.1,2.1,-1.1,-2.1,3.1]]]]).permute(0, 2, 1, 3)
print(t.shape)
print(t)
print(t.reshape(1, 3, 1, 12).shape)
print(t.reshape(1, 3, 1, 12))