import torch
import torch.nn as nn


class MultiTokenPrediction(nn.Module):
    """Módulo de previsão multi-token"""
    def __init__(self, dim, depth, vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim_feedforward=4*dim)
            for _ in range(depth)
        ])
        self.proj = nn.Linear(2*dim, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, h_prev, next_token_emb):
        h = torch.cat([h_prev, next_token_emb], dim=-1)
        h = self.proj(h)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
