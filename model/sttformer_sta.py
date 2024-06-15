import torch.nn as nn
import torch
import numpy as np
import math
#from .sta_block import STA_Block
from .EMAttention import EMA
#from .pos_embed import Pos_Embed
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


class STPos_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints, s_or_t="st"):
        super().__init__()
        allowed_strings = ["s", "t", "s+t", "st"]
        if s_or_t in allowed_strings:
            self.s_or_t = s_or_t
        else:
            raise ValueError("select from [\"s\", \"t\", \"s+t\", \"st\"]")
        # pe [120, 25, 64]
        if self.s_or_t == "st":
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
        else:
            pe = torch.zeros(num_frames, num_joints, channels)
            # div_term [32]
            div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))
            if "t" in self.s_or_t:
                tpos_list = [tk for tk in range(num_frames)]
                tposition = torch.from_numpy(np.array(tpos_list)).unsqueeze(1).float()
                pe[:, :, 0::2] += torch.sin(tposition * div_term).unsqueeze(1)
                pe[:, :, 1::2] += torch.cos(tposition * div_term).unsqueeze(1)
            if "s" in self.s_or_t:
                spos_list = [st for st in range(num_joints)]
                sposition = torch.from_numpy(np.array(spos_list)).unsqueeze(1).float()
                pe[:, :, 0::2] += torch.sin(sposition * div_term).unsqueeze(0)
                pe[:, :, 1::2] += torch.cos(sposition * div_term).unsqueeze(0)
            pe = pe.permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = self.pe[:, :, :x.size(2)]
        return x


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)
    
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    #  The elements' rotation angle\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    #  index t = [0, 1,..., joint_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [joint_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # m * \theta

    #  The result is a complex vector
    #  set freqs = [x, y]
    #  freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# RoPE
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
):
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Conversion to complex field
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # Apply the rotation operation, then turn the result back to the field of real numbers
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class Model(nn.Module):
    def __init__(self, len_parts, num_classes, num_joints,
                 num_frames, num_heads, num_persons, num_channels,
                 kernel_size, use_pes=True, config=None,
                 att_drop=0, dropout=0, dropout2d=0, factor=8, RoPE=False, pes_style="st"):
        super().__init__()

        self.len_parts = len_parts
        in_channels = config[0][0]
        self.out_channels = config[-1][1]
        num_frames = num_frames // len_parts
        num_joints = num_joints * len_parts

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1))

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(STA_Block(in_channels, out_channels, qkv_dim,
                                         num_frames=num_frames,
                                         num_joints=num_joints,
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop,
                                         factor=factor,
                                         RoPE=RoPE,
                                         pes_style=pes_style))

        self.fc = nn.Linear(self.out_channels, num_classes)
        #self.fc2 = nn.Linear(self.out_channels, 2)
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
        #return self.fc2(x)
        
        
class STA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0, factor=8, RoPE=False, pes_style="s+t"):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        self.RoPE = RoPE
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)

        # Spatio-Temporal Tuples Attention
        # if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        if self.use_pes: self.pes = STPos_Embed(in_channels, num_frames, num_joints, pes_style)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        #self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(EMA(out_channels, factor=factor), nn.BatchNorm2d(out_channels))


        # Inter-Frame Feature Aggregation
        self.out_nett = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)), nn.BatchNorm2d(out_channels))
        #self.out_nett2 = nn.Sequential(EMA(out_channels, factor=factor), nn.BatchNorm2d(out_channels))
        
        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)
    
    def forward(self, x):

        N, C, T, V = x.size()
        # Spatio-Temporal Tuples Attention
        xs = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        # q, k = [N, H, d, T, V]
        if self.RoPE:
            freqs_cis = precompute_freqs_cis(self.qkv_dim, V).cuda()
            q = q.permute(0, 1, 3, 4, 2).reshape(-1, V, self.qkv_dim)
            k = k.permute(0, 1, 3, 4, 2).reshape(-1, V, self.qkv_dim)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q = q.reshape(N, self.num_heads, self.qkv_dim, T, V)
            k = k.reshape(N, self.num_heads, self.qkv_dim, T, V)
        #attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas / 10
        attention = self.softmax(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        #attention = softmax_one(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T), -1) * self.alphas
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        # modify
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)

        # Inter-Frame Feature Aggregation
        #xt = self.relu(self.out_nett2(xs) + self.rest(xs))
        xt = self.relu(self.out_nett(xs) + self.rest(xs))
        return xt
