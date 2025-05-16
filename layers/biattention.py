import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys):
        B, L, H, E = queries.shape
        _, S, _, D = keys.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) #32,8,321,96
        A2 = self.dropout(torch.softmax(scale * scores.transpose(-1, -2), dim=-1))#32,8,96,321

        V1 = torch.einsum("bhls,bshd->blhd", A, keys)
        V2 = torch.einsum("bhls,bshd->blhd", A2, queries)

        if self.output_attention:
            return (V1.contiguous(), V2.contiguous(), A)
        else:
            return (V1.contiguous(), V2.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, enc_in, pred_len, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.dec = nn.Conv1d(d_model, pred_len, 1)  # d_model
        self.dec2 = nn.Linear(d_model, enc_in)
        self.out_projection1 = nn.Linear(d_keys * n_heads, d_model)
        self.out_projection2 = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries, keys):#, values):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queriesv = self.query_projection(queries).view(B, L, H, -1)
        keysv = self.key_projection(keys).view(B, S, H, -1)

        out1, out2, attn = self.inner_attention(
            queriesv,
            keysv,
        )
        out1 = out1.view(B, L, -1)
        out1 = queries + self.out_projection1(out1)
        out2 = out2.view(B, S, -1)
        out2 = keys + self.out_projection2(out2)
        out = self.dec(out1.permute(0, 2, 1)) + self.dec2(out2)
        return out, attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, queries, keys, values):
        new_x, attn = self.attention(
            queries, keys, values
        )
        x = queries + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn



class Encoder(nn.Module):
    def __init__(self, attn_layers, enc_in, seq_len, d_model, embed, freq, dropout, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        queries = self.q_embedding(x.permute(0, 2, 1))  # 32,321,512
        keys = values = self.kv_embedding(x)  # 32,96,512
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(queries, keys, values)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns