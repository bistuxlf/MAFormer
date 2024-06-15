import torch
import torch.nn as nn
import math
import numpy as np


class Pos_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints):
        super().__init__()

        pos_list = []
        for tk in range(num_frames):
            for st in range(num_joints):
                pos_list.append(st)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(num_frames * num_joints, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(num_frames, num_joints, channels).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = self.pe[:, :, :x.size(2)]
        return x


N = 2
C = 64
T = 20
V_node =25
V_part = 150
num_heads = 3
x = torch.randn([N, C, T, V_part])
pe_net = Pos_Embed(C, T, V_part)
# print(pe_net.pe)

### Centrality Encoding
all_degree = torch.tensor([4, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3,
                           3, 3, 2, 5, 2, 3, 2, 3])
in_degree_encoder = nn.Embedding(V_node, C, padding_idx=0)
degree_embedding = in_degree_encoder(all_degree.repeat(N, 1)) # N,25,C
degree_embedding = degree_embedding.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, 1, 6)
x = x + degree_embedding
# graph_token = nn.Embedding(1, C) # 整个图的embedding
# graph_token_feature = graph_token.weight.unsqueeze(0).repeat(N, 1, 1)
# graph_node_feature = torch.cat([graph_token_feature, x], dim=1)
print(degree_embedding.shape)
print(x.shape)
###

### Spatial Encoding 连通性和最短路径 node_group_feature 也可以作为补充特征，比如将一些节点在时间上聚合，作为一个group
                           #0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# 结构距离
Spatial_pos = torch.tensor([[0, 1, 3, 4, 3, 4, 5, 6, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 4, 2, 7, 7, 7, 7],
                           [1, 0, 2, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 1, 6, 6, 6, 6],
                           [3, 2, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 1, 6, 6, 6, 6],
                           [4, 3, 1, 0, 3, 4, 5, 6, 3, 4, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 2, 7, 7, 7, 7],
                           [3, 2, 2, 3, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 1, 4, 4, 6, 6],
                           [4, 3, 3, 4, 1, 0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 2, 3, 3, 7, 7],
                           [5, 4, 4, 5, 2, 1, 0, 1, 4, 5, 6, 7, 6, 7, 8, 9, 6, 7, 8, 9, 3, 2, 2, 8, 8],
                           [6, 5, 5, 6, 3, 2, 1, 0, 5, 6, 7, 8, 7, 8, 9, 10,7, 8, 9, 10,4, 1, 1, 9, 9],
                           [3, 2, 2, 3, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 1, 6, 6, 4, 4],
                           [4, 3, 3, 4, 3, 4, 5, 6, 4, 0, 1, 2, 5, 6, 7, 8, 5, 6, 7, 8, 2, 7, 7, 3, 3],
                           [5, 4, 4, 5, 4, 5, 6, 7, 2, 1, 0, 1, 6, 7, 8, 9, 6, 7, 8, 9, 3, 8, 8, 2, 2],
                           [6, 5, 5, 6, 5, 6, 7, 8, 3, 2, 1, 0, 7, 8, 9,10, 7, 8, 9,10, 4, 9, 9, 1, 1],
                           [1, 2, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 0, 1, 2, 3, 2, 3, 4, 5, 3, 8, 8, 8, 8],
                           [2, 3, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 1, 0, 1, 2, 3, 4, 5, 6, 4, 9, 9, 9, 9],
                           [3, 4, 6, 7, 6, 7, 8, 9, 6, 7, 8, 9, 2, 1, 0, 1, 4, 5, 6, 7, 5,10,10,10,10],
                           [4, 5, 7, 8, 7, 8, 9,10, 7, 8, 9,10, 3, 2, 1, 0, 5, 6, 7, 8, 6,11,11,11,11],
                           [1, 2, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 2, 3, 4, 5, 0, 1, 2, 3, 3, 8, 8, 8, 8],
                           [2, 3, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 4, 5, 6, 1, 1, 0, 1, 2, 4, 9, 9, 9, 9],
                           [3, 4, 6, 7, 6, 7, 8, 9, 6, 7, 8, 9, 4, 5, 6, 7, 2, 1, 0, 1, 5,10,10,10,10],
                           [4, 5, 7, 8, 7, 8, 9,10, 7, 8, 9,10, 5, 6, 7, 8, 3, 2, 1, 0, 6,11,11,11,11],
                           [2, 1, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 3, 4, 5, 6, 0, 5, 5, 5, 5],
                           [7, 6, 6, 7, 4, 3, 2, 1, 6, 7, 8, 9, 8, 9,10,11, 8, 9,10,11, 5, 0, 2,10,10],
                           [7, 6, 6, 7, 4, 3, 2, 1, 6, 7, 8, 9, 8, 9,10,11, 8, 9,10,11, 5, 2, 0,10,10],
                           [7, 6, 6, 7, 6, 7, 8, 9, 4, 3, 2, 1, 8, 9,10,11, 8, 9,10,11, 5,10,10, 0,10],
                           [7, 6, 6, 7, 6, 7, 8, 9, 4, 3, 2, 1, 8, 9,10,11, 8, 9,10,11, 5,10,10,10, 0]])
# 实际距离?

with torch.no_grad():
    SSpatial_pos = torch.zeros(150, 150,dtype=int)
    for i in range(6):
        i = i * 25
        SSpatial_pos[i:i + 25, i:i + 25] = Spatial_pos
print(SSpatial_pos)

print(Spatial_pos.shape)

Spatial_pos_encoder = nn.Embedding(150, num_heads, padding_idx=0)
Spatial_pos_embedding = Spatial_pos_encoder(SSpatial_pos.repeat(N,1,1)).permute(0, 3, 1, 2)
attention_bias = torch.ones(N, num_heads, 150, 150)
attention_bias = attention_bias + Spatial_pos_embedding
print(Spatial_pos_embedding.shape)
###

