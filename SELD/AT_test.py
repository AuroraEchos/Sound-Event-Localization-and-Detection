import torch
import torch.nn as nn

import math
from dual_quaternion.dual_quaternion_layers import * 
from quaternion.quaternion_layers import *

torch.backends.cudnn.enabled = False

class ATLayer(nn.Module):
    def __init__(self, in_channels, key_size=8, value_size=16):
        super(ATLayer, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.linear_output = nn.Linear(value_size, in_channels)
        self.sqrt_key_size = math.sqrt(key_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        # input shape: (N, in_channels, T)
        N, in_channels, T = input.size()
        
        # Generating causal mask for attention
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(input.device)
        
        # Permute to shape (N, T, in_channels)
        input = input.permute(0, 2, 1)
        
        # Linear projections
        keys = self.linear_keys(input)    # (N, T, key_size)
        query = self.linear_query(input)  # (N, T, key_size)
        values = self.linear_values(input) # (N, T, value_size)

        # Scaled dot-product attention
        attention_scores = torch.bmm(query, keys.transpose(1, 2)) / self.sqrt_key_size  # (N, T, T)
        attention_scores.masked_fill_(mask, -float('inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, T, T)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        attention_output = torch.bmm(attention_weights, values)  # (N, T, value_size)
        attention_output = self.linear_output(attention_output).permute(0, 2, 1)  # (N, in_channels, T)

        return attention_output, attention_weights

class ConvATLayer(nn.Module):
    def __init__(self, in_channels, key_size=8, value_size=16):
        super(ConvATLayer, self).__init__()
        self.conv_query = nn.Conv1d(in_channels, key_size, kernel_size=1)
        self.conv_keys = nn.Conv1d(in_channels, key_size, kernel_size=1)
        self.conv_values = nn.Conv1d(in_channels, value_size, kernel_size=1)
        self.conv_output = nn.Conv1d(value_size, in_channels, kernel_size=1)
        self.sqrt_key_size = math.sqrt(key_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        # input shape: (N, in_channels, T)
        N, in_channels, T = input.size()
        
        # Generating causal mask for attention
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(input.device)
        
        # Linear projections using conv1d (keeping channel dimension)
        keys = self.conv_keys(input)    # (N, key_size, T)
        query = self.conv_query(input)  # (N, key_size, T)
        values = self.conv_values(input) # (N, value_size, T)

        # Permute to shape (N, T, key_size) for batch matrix multiplication
        keys = keys.permute(0, 2, 1)  # (N, T, key_size)
        query = query.permute(0, 2, 1)  # (N, T, key_size)
        values = values.permute(0, 2, 1)  # (N, T, value_size)

        # Scaled dot-product attention
        attention_scores = torch.bmm(query, keys.transpose(1, 2)) / self.sqrt_key_size  # (N, T, T)
        attention_scores.masked_fill_(mask, -float('inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, T, T)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        attention_output = torch.bmm(attention_weights, values)  # (N, T, value_size)
        attention_output = self.conv_output(attention_output.permute(0, 2, 1))  # (N, in_channels, T)

        return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split the embedding into num_heads different pieces
        values = self.values(values).view(N, value_len, self.num_heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.num_heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.num_heads, self.head_dim)
        
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, -1)
        out = self.fc_out(out)
        return out

class ConvMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(ConvMultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.values = nn.Conv1d(embed_size, embed_size, kernel_size=1, bias=False)
        self.keys = nn.Conv1d(embed_size, embed_size, kernel_size=1, bias=False)
        self.queries = nn.Conv1d(embed_size, embed_size, kernel_size=1, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, v, k, q, mask=None):
        N = q.shape[0]
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]
        
        # Transpose to (batch_size, embed_size, seq_length) for Conv1d
        v = v.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        q = q.permute(0, 2, 1)
        
        # Apply convolution
        v = self.values(v).view(N, self.head_dim * self.num_heads, value_len).permute(0, 2, 1).view(N, value_len, self.num_heads, self.head_dim)
        k = self.keys(k).view(N, self.head_dim * self.num_heads, key_len).permute(0, 2, 1).view(N, key_len, self.num_heads, self.head_dim)
        q = self.queries(q).view(N, self.head_dim * self.num_heads, query_len).permute(0, 2, 1).view(N, query_len, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        
        if mask is not None:
            # Apply mask to the energy tensor
            energy = energy.masked_fill(mask == 0, float("-1e9"))
        
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(N, query_len, -1)
        out = self.fc_out(out)
        
        return out
    

if __name__ == '__main__':
    N, in_channels, T = 2, 32, 10
    key_size = 8
    value_size = 16
    input = torch.randn(N, in_channels, T)

    # Attention layer
    at_layer = ATLayer(in_channels, key_size, value_size)
    output, weights = at_layer(input)
    print('Attention Layer')
    print(input.shape)
    print(output.shape)

    # Convolutional Attention layer
    conv_at_layer = ConvATLayer(in_channels, key_size, value_size)
    output, weights = conv_at_layer(input)
    print('Convolutional Attention Layer')
    print(input.shape)
    print(output.shape)

    # Multi-head attention
    embed_size = 256
    num_heads = 8
    batch_size = 64
    seq_length = 50
    x = torch.rand((batch_size, seq_length, embed_size))
    mask = None
    mha = MultiHeadAttention(embed_size, num_heads)
    output = mha(x, x, x, mask)
    print('Multi-head Attention')
    print(x.shape)
    print(output.shape)

    # Convolutional Multi-head attention
    embed_size = 256
    num_heads = 8
    batch_size = 64
    seq_length = 50
    x = torch.rand((batch_size, seq_length, embed_size))
    mask = None
    mha = ConvMultiHeadAttention(embed_size, num_heads)
    output = mha(x, x, x, mask)
    print('Convolutional Multi-head Attention')
    print(x.shape)
    print(output.shape)


