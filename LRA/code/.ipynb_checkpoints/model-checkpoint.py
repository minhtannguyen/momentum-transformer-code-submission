
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from attention import Attention

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )

    def forward(self, X, mask):
        X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = Transformer(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", Transformer(config))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    def forward(self, input_ids, mask = None):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                X = self.transformer(X, mask)
        else:
            for idx in range(self.num_layers):
                X = getattr(self, f"transformer_{idx}")(X, mask)

        X = self.norm(X) * mask[:, :, None]

        return X
    
class TransformerAdaptive(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )
        
        self.res_stepsize = config["res_stepsize"]
        self.res_delta = config["res_delta"]

    def forward(self, X, v, gradval, mask):
        gradvaly = self.dropout1(self.mha(self.norm1(X), mask))
        
        if torch.all(v == torch.zeros_like(X)):
            res_mu = (1.0 - torch.sqrt(torch.norm(gradvaly - gradval, dim=2, keepdim=True)/torch.norm(gradval, dim=2, keepdim=True)))**2
        else:
            res_mu = (1.0 - torch.sqrt(torch.norm(gradvaly - gradval, dim=2, keepdim=True)/torch.norm(v, dim=2, keepdim=True)))**2
        
        res_mu = torch.clamp(res_mu, min=0.0, max=1.0 - self.res_delta)
        vy = res_mu * v - gradvaly
        
        X = X - self.res_stepsize * vy
        X = self.mlpblock(self.norm2(X)) + X
        return X, vy, gradvaly

class ModelAdaptive(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = TransformerAdaptive(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", TransformerAdaptive(config))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    def forward(self, input_ids, mask = None):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            v = torch.zeros_like(X)
            gradval = torch.ones_like(X)
            for idx in range(self.num_layers):
                X, v, gradval = self.transformer(X, v, gradval, mask)
        else:
            v = torch.zeros_like(X)
            gradval = torch.ones_like(X)
            for idx in range(self.num_layers):
                X, v, gradval = getattr(self, f"transformer_{idx}")(X, v, gradval, mask)

        X = self.norm(X) * mask[:, :, None]

        return X
    
class TransformerMomentum(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )
        
        self.res_stepsize = config["res_stepsize"]
        self.res_mu = config["res_mu"]

    def forward(self, X, v, mask):
        gradvaly = self.dropout1(self.mha(self.norm1(X), mask))
        vy = self.res_mu * v - gradvaly
        
        X = X - self.res_stepsize * vy
        X = self.mlpblock(self.norm2(X)) + X
        return X, vy
    
class ModelMomentum(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = TransformerMomentum(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", TransformerMomentum(config))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    def forward(self, input_ids, mask = None):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            v = torch.zeros_like(X)
            for idx in range(self.num_layers):
                X, v = self.transformer(X, v, mask)
        else:
            v = torch.zeros_like(X)
            for idx in range(self.num_layers):
                X, v = getattr(self, f"transformer_{idx}")(X, v, mask)

        X = self.norm(X) * mask[:, :, None]

        return X