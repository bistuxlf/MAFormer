import torch.nn as nn
import torch
import numpy as np
from .sta_block import GAT_Block

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class get_relative_map(nn.Module):
    def __init__(self, in_channels, num_classes, num_joints):
        super(get_relative_map, self).__init__()
        # 在V维度上进行pooling
        self.embedding = nn.Conv2d(in_channels=num_classes, out_channels=in_channels, kernel_size=(1, 1))
        self.pool = nn.AvgPool2d(kernel_size=(1, num_joints))
        # 两个不同参数的1*1卷积
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels/4, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels/4, kernel_size=(1, 1))

    def forward(self, x):
        x = self.embedding(x)
        # 在V维度上进行pooling，变成（N , C=64, T=120, 1）
        x = self.pool(x)
        # 分别通过两个不同参数的1*1卷积变成得到两组f1=（N , C=16, T=120, 1），f2=（N , C=16, T=120, 1）
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        # 将f1与f2的转置进行矩阵相乘，得到（N , C=16, T=120, T=120）
        f2T = f2.transpose(2, 3)
        x = torch.matmul(f1, f2T)
        x = torch.mean(torch.mean(x, dim=0), dim=0)
        x = torch.softmax(x, dim=-1)
        # 对其进行两次平均池化变为（T=120, T=120）
        return x.squeeze()


class Model(nn.Module):
    def __init__(self, len_parts, num_classes, num_joints,
                 num_frames, num_heads, num_persons, num_channels,
                 kernel_size, use_pes=True, config=None,
                 att_drop=0, dropout=0, dropout2d=0, graph=None, graph_args=dict()):
        super().__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        with torch.no_grad():
            self.A = self.graph.A
        self.len_parts = len_parts
        in_channels = config[0][0]
        self.out_channels = config[-1][1]
        self.num_frames = num_frames // len_parts
        self.num_joints = num_joints * len_parts
        self.relative = get_relative_map(in_channels, num_classes, num_joints)
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1))

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(GAT_Block(in_channels, out_channels, qkv_dim,
                                         num_frames=self.num_frames,
                                         num_joints=self.num_joints,
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop,
                                         A=self.A))

        self.fc = nn.Linear(self.out_channels, num_classes)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        N, C, T, V, M = x.shape
        with torch.no_grad():
            relative = self.get_relative_map(x)
            node_groups = [[] for _ in range(self.len_parts)]
            for i in range(self.len_parts):
                node_groups[i].append(i)
            # 对剩下的节点进行分组
            for i in range(self.len_parts, T):
                max_sum = -float('inf')
                max_group = -1
                for j in range(self.len_parts):
                    if len(node_groups[j]) == T / self.len_parts:
                        continue
                    group_sum = 0
                    for node_idx in node_groups[j]:
                        group_sum += relative[i][node_idx] + relative[node_idx][i]
                    if group_sum > max_sum:
                        max_sum = group_sum
                        max_group = j
                node_groups[max_group].append(i)
            node_groups = torch.tensor(node_groups)
        x = torch.index_select(x, dim=2, index=node_groups.reshape(-1))
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)
        x = self.input_map(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
        # NM, C, T, V
        x = x.view(N, M, self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)