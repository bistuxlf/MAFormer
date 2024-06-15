import torch
import torch.nn as nn
import numpy as np
from .pos_embed import Pos_Embed
#

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


class STA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads, len_parts,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)

        # Spatio-Temporal Tuples Attention
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.all_degree = torch.tensor([4, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3,
                                   3, 3, 2, 5, 2, 3, 2, 3])
        self.degree_encoder = nn.Embedding(num_joints, in_channels, padding_idx=0)
        self.Spatial_pos_encoder = nn.Embedding(num_joints, num_heads, padding_idx=0)
        with torch.no_grad():
            self.Spatial_pos = torch.zeros(num_joints, num_joints, dtype=int)
            for i in range(6):
                i = i * 25
                self.Spatial_pos[i:i + 25, i:i + 25] = Spatial_pos
            self.Spatial_pos = self.Spatial_pos.cuda()
        self.len_parts = len_parts
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

        # Inter-Frame Feature Aggregation
        self.out_nett = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
                                      nn.BatchNorm2d(out_channels))

        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):

        N, C, T, V = x.size()
        # Spatio-Temporal Tuples Attention
        xs = self.pes(x) + x if self.use_pes else x
        degree_embedding = self.degree_encoder(self.all_degree.repeat(N, 1))
        degree_embedding = degree_embedding.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, 1, self.len_parts)
        xs = xs + degree_embedding
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        Spatial_embedding = self.Spatial_pos_encoder(self.Spatial_pos.repeat(N,1,1)).permute(0, 3, 1, 2)
        attention = attention + self.att0s.repeat(N, 1, 1, 1) + Spatial_embedding  # N, 3, 150, 150
        attention = self.drop(attention)
        # modify
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T,
                                                                                V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)

        # Inter-Frame Feature Aggregation
        xt = self.relu(self.out_nett(xs) + self.rest(xs))

        return xt

