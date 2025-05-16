import torch
import torch.nn as nn
from math import sqrt
from layers.Embed import PositionalEmbedding

def norm(x):
    mean = x.mean(1, keepdim=True).detach()  # B x 1 x E
    x = x - mean
    std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
    x = x / std
    return x, mean, std

def denorm(x, mean, std):
    x = x * std + mean
    return x

class PWattn(nn.Module):#Point-Wise Skip-Attention
    def __init__(self, seq_len, pred_len, n_patches=16, attention_dropout=0.0):
        super(PWattn, self).__init__()
        self.query_projection = nn.Linear(seq_len, pred_len)
        self.key_projection = nn.Linear(seq_len, seq_len)
        self.value_projection = nn.Linear(seq_len, seq_len)
        self.out_projection = nn.Linear(pred_len, pred_len)
        self.dropout = nn.Dropout(attention_dropout)
        self.P = n_patches
        self.seq_len = seq_len
        self.pred_len = pred_len
    def forward(self, x):
        P = self.P
        B = x.shape[0]
        query = torch.relu(self.query_projection(x))
        queries = query.view(B, self.pred_len // P, P)
        keys = self.key_projection(x).view(B, self.seq_len // P,  P)
        values = torch.relu(self.value_projection(x)).view(B, self.seq_len // P,  P)
        scores = torch.einsum("ble,bse->bls", queries, keys)
        A = self.dropout(torch.softmax(scores / sqrt(P), dim=-1))
        V = torch.einsum("bls,bse->ble", A, values).contiguous()
        V = V.view(B, self.pred_len)
        V = query + self.out_projection(V)
        return V
    
class GAttn(nn.Module):#Granger-Attention
    def __init__(self, d_model, n_heads, top_k=2, attention_dropout=0.0):
        super(GAttn, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.active = nn.Softmax(dim=-1)
        self.top_k = top_k
    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        H = self.n_heads
        _, S, _ = keys.shape
        queries = torch.relu(self.query_projection(queries)).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = torch.relu(self.value_projection(values)).view(B, S, H, -1)
        E = queries.shape[-1]
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        filter_value = -float('Inf')
        indices_to_remove = scores < torch.topk(scores, self.top_k)[0][..., -1, None]
        scores[indices_to_remove] = filter_value
        A = self.dropout(self.active(scale * scores))
        V = torch.einsum("bhls,bshd->blhd", A, values).contiguous()
        out = V.view(B, L, -1)
        return self.out_projection(out), A
    
class Encoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, top_k=10):#8
        super(Encoder, self).__init__()
        self.attn_layer1 = GAttn(d_model, n_heads, top_k)
    def forward(self, x):
        new_x, attns = self.attn_layer1(x, x, x)
        x = x + new_x
        return x, attns
    
class GFAN(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, pred_len, n_patches, top_k=2, alpha=0.5, dropout=0.0):
        super(GFAN, self).__init__()
        self.enc = nn.Linear(d_model, d_model)
        self.ac = nn.LeakyReLU(alpha)
        self.encoder = Encoder(d_model, n_heads, top_k)
        self.ff1 = PWattn(d_model, d_ff, n_patches)
        self.ff2 = PWattn(d_ff, d_model, n_patches)#pred_len#
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_ff)
        self.activation = nn.ELU()
    def forward(self, x, n_vars):
        # GAttn
        y, attn = self.encoder(x)
        y = self.enc(self.ac(y))
        # point-wise skip-attn
        y = torch.reshape(y, (y.shape[0] * y.shape[1], y.shape[2]))
        y = self.dropout(self.norm2(self.activation(self.ff1(y))))
        y = self.dropout(self.ff2(y))
        y = torch.reshape(y, (-1, n_vars, y.shape[-1]))
        return y, attn
    
class Embedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(seq_len, d_model))
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.W_pos = PositionalEmbedding(d_model=d_model)
    def forward(self, x):
        x = torch.relu(torch.matmul(x, self.weight)) + self.W_pos(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention
        self.fan = GFAN(configs.d_model, configs.d_ff, configs.n_heads, configs.pred_len, configs.n_patches, configs.k_gran, configs.alpha, configs.dropout)
        self.embed = Embedding(configs.seq_len, configs.d_model)
        if configs.is_linear:
            self.fc = nn.Linear(configs.d_model, configs.pred_len)
        self.is_fc = configs.is_linear
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        n_vars = x_enc.shape[-1]
        x_enc, mean, std = norm(x_enc)
        x_enc = x_enc.permute(0, 2, 1)
        enc_in = self.embed(x_enc)
        dec_out, attn = self.fan(enc_in, n_vars)
        if self.is_fc:
            dec_out = self.fc(dec_out)#
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = denorm(dec_out, mean, std)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attn
        else:
            return dec_out[:, -self.pred_len:, :]