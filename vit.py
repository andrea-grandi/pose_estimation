import torch
import torch.nn as nn
import math


class PatchEmbeddings(nn.Module):
    """
    Patch Embeddings:
    
    aim: transform the imput image into patches embeddings
    """

    def __init__(self, patch_size, img_size, in_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_dim, stride=patch_dim)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PositionEncoding(nn.Module):
    """
    Positional Encoding:

    aim: add the positional encodings to the input embeddings
    """

    def __init__(self, d_model, num_patches):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches

        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))

    def forward(self, x):
        return x + self.position_encoding


class LayerNorm(nn.Module):
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std - self.eps) + self.bias


class FeedForward(nn.Module):
    
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.dropout(self.ff(x))


