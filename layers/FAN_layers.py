import torch
import torch.nn as nn
from math import sqrt
from layers.Embed import PositionalEmbedding
class PWattn(nn.Module):#Point-Wise-Attention
    def __init__(self, seq_len, pred_len, n_heads=16, attention_dropout=0.0):#traf:16 8,0.1
        super(PWattn, self).__init__()
        self.query_projection = nn.Linear(seq_len, pred_len)
        self.key_projection = nn.Linear(seq_len, seq_len)
        self.value_projection = nn.Linear(seq_len, seq_len)
        self.out_projection = nn.Linear(pred_len, pred_len)
        self.dropout = nn.Dropout(attention_dropout)
        self.H = n_heads
        self.seq_len = seq_len
        self.pred_len = pred_len
    def forward(self, x):
        H = self.H
        B = x.shape[0]
        query = self.query_projection(x)
        queries = query.view(B, self.pred_len // H, H)
        keys = self.key_projection(x).view(B, self.seq_len // H,  H)
        values = self.value_projection(x).view(B, self.seq_len // H,  H)
        scores = torch.einsum("ble,bse->bls", queries, keys)
        A = self.dropout(torch.softmax(scores / sqrt(H), dim=-1))
        V = torch.einsum("bls,bse->ble", A, values).contiguous()
        V = V.view(B, self.pred_len)
        V = query + self.out_projection(V)
        return V

class CAttn(nn.Module):#Channel-Wise-Attention
    def __init__(self, d_model, n_heads, attention_dropout=0.0, activation='softmax', alpha=0.5):#0.5 0.3 0.5 0.1
        super(CAttn, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)#d_model
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.active = nn.Softmax(dim=-1) if activation == "softmax" else nn.LeakyReLU(alpha)#0.3
    def forward(self, queries):
        B, L, _ = queries.shape
        H = self.n_heads
        keys = values = queries
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        E = queries.shape[-1]
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(self.active(scale * scores))
        V = torch.einsum("bhls,bshd->blhd", A, values).contiguous()
        out = V.view(B, L, -1)
        return self.out_projection(out), A
class Encoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn_layer1 = CAttn(d_model, n_heads, activation='softmax')#leakyrelu softmax
        self.attn_layer2 = CAttn(d_model, n_heads, activation='leakyrelu')
    def forward(self, x):
        attns = []
        x, attn1 = self.attn_layer1(x)
        attns.append(attn1)
        x, attn2 = self.attn_layer2(x)
        attns.append(attn2)
        return x#, attns
class FAN_layer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, pred_len, dropout=0.0):#relu , pred_len
        super(FAN_layer, self).__init__()
        self.encoder = Encoder(d_model, n_heads)
        self.ff1 = PWattn(d_model, d_ff)
        self.ff2 = PWattn(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_ff)
        self.activation = nn.ELU()#GELU()
        #self.fc = nn.Linear(d_model, pred_len)
    def forward(self, x, n_vars):
        new_x = self.encoder(x)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = torch.reshape(y, (y.shape[0] * y.shape[1], y.shape[2]))
        y = self.dropout(self.norm2(self.activation(self.ff1(y))))
        y = self.dropout(self.ff2(y))
        y = torch.reshape(y, (-1, n_vars, y.shape[-1]))
        #out = self.fc(y)
        return y#out
class Embedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(seq_len, d_model))
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.W_pos = PositionalEmbedding(d_model=d_model)
    def forward(self, x):
        x = torch.matmul(x, self.weight) + self.W_pos(x)
        return x
class FAN_backbone(nn.Module):
    def __init__(self, seq_len, pred_len, n_heads, d_model, d_ff, layers, output_attention):
        super(FAN_backbone, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.layer = layers
        #self.emb = Embedding(seq_len, d_model)
        e_layers = [Embedding(seq_len, d_model)]
        for i in range(layers - 1):
            e_layers += [Embedding(d_model, d_model)]
        self.emb_layers = nn.Sequential(*e_layers)
        f_layers = [FAN_layer(d_model, d_ff, n_heads, pred_len)]
        for i in range(layers - 1):
            f_layers += [FAN_layer(d_model, d_ff, n_heads, pred_len)]
        self.fan_layers = nn.Sequential(*f_layers)
        self.fc = nn.Linear(d_model, pred_len)
    def forward(self, x, n_vars):
        #x = self.emb(x)
        for i in range(self.layer):
            x = self.emb_layers[i](x)
            x = x + self.fan_layers[i](x, n_vars)
        x = self.fc(x)
        return x#dec_out