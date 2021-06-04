
import torch
import torch.nn as nn
import math
import numpy as np

class MomentumAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mu = config["mu"]
        self.stepsize = config["stepsize"]
        

    def forward(self, Q, K, V, mask):
        Q = (nn.functional.elu(Q) + 1) / math.sqrt(math.sqrt(Q.size(2)))
        K = (nn.functional.elu(K) + 1) * mask[:, None, :, None] / math.sqrt(math.sqrt(K.size(2)))
        V = V * mask[:, None, :, None]
        
        L = K.shape[2]
        momentum_weight = (self.stepsize * (1.0 - torch.pow(torch.ones(L) * self.mu, L - torch.from_numpy(np.arange(1,L+1)) + 1))/(1.0 - self.mu)).to(K)
        K = K * momentum_weight[None, None, :, None]

        X = torch.matmul(Q, torch.matmul(torch.transpose(K, -2, -1), V))

        return X
