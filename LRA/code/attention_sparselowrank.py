
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseLowrankAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.diag_size = config["diag_size"]
        self.num_head = config["num_head"]
        
        self.kernels = config["kernels"]
        self.rank_k = len(self.kernels)
        
        if config["sparse_ratio"] < 1.0: # e.g. sparse_ratio = 0.5
            self.type_blend = 0
        elif config["sparse_ratio"] < 2.0: # e.g. sparse_ratio = 1.5
            self.type_blend = 1
        elif config["sparse_ratio"] < 3.0: # e.g. sparse_ratio = 2.5
            self.type_blend = 2
        elif config["sparse_ratio"] < 4.0:  # e.g. sparse_ratio = 3.5
            self.type_blend = 3
        elif config["sparse_ratio"] < 5.0: # e.g. sparse_ratio = 4.5
            self.type_blend = 4 
        elif config["sparse_ratio"] < 6.0: # e.g. sparse_ratio = 5.5
            self.type_blend = 5 
        elif config["sparse_ratio"] < 7.0: # e.g. sparse_ratio = 6.5
            self.type_blend = 6 # sparse only
        elif config["sparse_ratio"] < 8.0: # e.g. sparse_ratio = 7.5
            self.type_blend = 7 # lowrank only 
        elif config["sparse_ratio"] < 9.0: # e.g. sparse_ratio = 8.5
            self.type_blend = 8 
        elif config["sparse_ratio"] < 10.0: # e.g. sparse_ratio = 9.5
            self.type_blend = 9 
        elif config["sparse_ratio"] < 11.0: # e.g. sparse_ratio = 10.5
            self.type_blend = 10 
        elif config["sparse_ratio"] < 12.0: # e.g. sparse_ratio = 11.5
            self.type_blend = 11 
        elif config["sparse_ratio"] < 13.0: # e.g. sparse_ratio = 12.5
            self.type_blend = 12 
        elif config["sparse_ratio"] < 14.0: # e.g. sparse_ratio = 13.5
            self.type_blend = 13
        elif config["sparse_ratio"] < 15.0: # e.g. sparse_ratio = 14.5
            self.type_blend = 14 
        elif config["sparse_ratio"] < 16.0: # e.g. sparse_ratio = 15.5
            self.type_blend = 15 
        elif config["sparse_ratio"] < 17.0: # e.g. sparse_ratio = 16.5
            self.type_blend = 16
        elif config["sparse_ratio"] < 18.0: # e.g. sparse_ratio = 17.5
            self.type_blend = 17
        elif config["sparse_ratio"] < 19.0: # e.g. sparse_ratio = 18.5
            self.type_blend = 18
        elif config["sparse_ratio"] < 20.0: # e.g. sparse_ratio = 19.5
            self.type_blend = 19
            
        if self.type_blend == 0:
            self.sparse_ratio = nn.Parameter(torch.Tensor([0.5]))
        elif self.type_blend == 1:
            self.sparse_ratio = nn.Parameter(torch.Tensor([0.5]))
        elif self.type_blend == 2:
            self.sparse_ratio = nn.Parameter(torch.Tensor([0.5]))
        elif self.type_blend == 3:
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, self.head_dim))
        elif self.type_blend == 4:  
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, self.head_dim))
        elif self.type_blend == 5:  
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, self.head_dim))
        elif self.type_blend == 6:
            self.sparse_ratio = 1.0 # sparse only
        elif self.type_blend == 7:
            self.sparse_ratio = 0.0 # lowrank only
        elif self.type_blend == 8:
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, self.head_dim))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([0.5]))
            self.lowrank_ratio2 = nn.Parameter(torch.Tensor([0.5]))
        elif self.type_blend == 9:
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, self.head_dim))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([0.5]))
            self.lowrank_ratio2 = nn.Parameter(torch.Tensor([0.5]))
        elif self.type_blend == 10:
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, 1))
            self.sparse_ratio2 = nn.Parameter(torch.ones(1, self.num_head, 1, 1))
        elif self.type_blend == 11:
            self.sparse_ratio = nn.Parameter(torch.ones(1, self.num_head, 1, 1))
            self.sparse_ratio2 = nn.Parameter(torch.zeros(1, self.num_head, 1, 1))
        elif self.type_blend == 12:
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, 1))
        elif self.type_blend == 13:
            self.sparse_ratio = nn.Parameter(torch.ones(1, self.num_head, 1, 1))
            self.sparse_ratio2 = nn.Parameter(torch.zeros(1, self.num_head, 1, 1))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([1.0]))
            self.lowrank_ratio2 = nn.Parameter(torch.Tensor([0.0]))
        elif self.type_blend == 14:
            self.sparse_ratio = nn.Parameter(torch.Tensor([0.5]))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([1.0]))
            self.lowrank_ratio2 = nn.Parameter(torch.Tensor([0.0]))
        elif self.type_blend == 15:
            self.sparse_ratio = nn.Parameter(torch.Tensor([0.5]))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([0.0]))
        elif self.type_blend == 16:  
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, self.head_dim))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([1.0]))
            self.lowrank_ratio2 = nn.Parameter(torch.Tensor([0.0]))
        elif self.type_blend == 17:  
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, self.head_dim))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([0.0]))
        elif self.type_blend == 18:
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, 1))
        elif self.type_blend == 19:
            self.sparse_ratio = nn.Parameter(torch.zeros(1, self.num_head, 1, 1))
            self.lowrank_ratio = nn.Parameter(torch.Tensor([1.0]))
            self.lowrank_ratio2 = nn.Parameter(torch.Tensor([0.0]))
            
        
    def forward(self, Q, K, V, mask):
        if self.type_blend != 7:
            attn_vec_sparse = self._forward_sparse(Q, K, V, mask) # batchsize x num_head x seq.len. x head_dim
        
        if self.type_blend != 6:
            attn_vec_lowrank = self._forward_lowrank(Q, K, V, mask) # batchsize x num_head x seq.len. x head_dim
            
        if self.type_blend == 0:
            X = self.sparse_ratio * attn_vec_sparse + attn_vec_lowrank
        elif self.type_blend == 1:
            X = attn_vec_sparse + self.sparse_ratio * attn_vec_lowrank
        elif self.type_blend == 2:
            X = self.sparse_ratio * attn_vec_sparse + (1.0 - self.sparse_ratio) * attn_vec_lowrank
        elif self.type_blend == 3:
            X = self.sparse_ratio * attn_vec_sparse + attn_vec_lowrank
        elif self.type_blend == 4:
            X = attn_vec_sparse + self.sparse_ratio * attn_vec_lowrank
        elif self.type_blend == 5:
            X = self.sparse_ratio * attn_vec_sparse + (1.0 - self.sparse_ratio) * attn_vec_lowrank
        elif self.type_blend == 6:
            X = attn_vec_sparse
        elif self.type_blend == 7:
            X = attn_vec_lowrank
        elif self.type_blend == 8:
            X = self.sparse_ratio * attn_vec_sparse + attn_vec_lowrank
        elif self.type_blend == 9:
            X = attn_vec_sparse + self.sparse_ratio * attn_vec_lowrank
        elif self.type_blend == 10:
            X = self.sparse_ratio * attn_vec_sparse + self.sparse_ratio2 * attn_vec_lowrank
        elif self.type_blend == 11:
            X = self.sparse_ratio * attn_vec_sparse + self.sparse_ratio2 * attn_vec_lowrank
        elif self.type_blend == 12:
            X = attn_vec_sparse + self.sparse_ratio * attn_vec_lowrank
        elif self.type_blend == 13:
            X = self.sparse_ratio * attn_vec_sparse + self.sparse_ratio2 * attn_vec_lowrank
        elif self.type_blend == 14:
            X = self.sparse_ratio * attn_vec_sparse + (1.0 - self.sparse_ratio) * attn_vec_lowrank
        elif self.type_blend == 15:
            X = self.sparse_ratio * attn_vec_sparse + (1.0 - self.sparse_ratio) * attn_vec_lowrank
        elif self.type_blend == 16:
            X = attn_vec_sparse + self.sparse_ratio * attn_vec_lowrank
        elif self.type_blend == 17:
            X = attn_vec_sparse + self.sparse_ratio * attn_vec_lowrank
        elif self.type_blend == 18:
            X = self.sparse_ratio * attn_vec_sparse + attn_vec_lowrank
        elif self.type_blend == 19:
            X = self.sparse_ratio * attn_vec_sparse + attn_vec_lowrank
        
        return X
    
    def _project_features(self, features, kernel_name):
        if kernel_name == 'elu':
            out = F.elu(features, 1., False) + 1.
        elif kernel_name == 'tanh':
            out = F.tanh(features) + 1.
        elif kernel_name == 'relu':
            out = F.relu(features, False)
        elif kernel_name == 'celu':
            out = F.celu(features, 1., False) + 1.
        elif kernel_name == 'sigmoid':
            out = F.sigmoid(features)
        elif kernel_name == 'leaky_relu':
            out = F.leaky_relu(features) + 1.
        elif kernel_name == 'softplus':
            out = F.softplus(features)
        elif kernel_name == 'tanh_orthogonal':
            out = 1. - F.tanh(features)
        elif kernel_name == 'elu_flip':
            out = F.elu(-features, 1., False) + 1.
        else:
            out = features
            
        return out
            
    def _forward_lowrank(self, Q, K, V, mask):
        V = V * mask[:, None, :, None]
        
        Q1 = self._project_features(Q, kernel_name=self.kernels[0]) / math.sqrt(math.sqrt(Q.size(2)))
        K1 = self._project_features(K, kernel_name=self.kernels[0]) * mask[:, None, :, None] / math.sqrt(math.sqrt(K.size(2)))
        X = torch.matmul(Q1, torch.matmul(torch.transpose(K1, -2, -1), V))
        
        if self.rank_k > 1:
            Q2 = self._project_features(Q, kernel_name=self.kernels[1]) / math.sqrt(math.sqrt(Q.size(2)))
            K2 = self._project_features(K, kernel_name=self.kernels[1]) * mask[:, None, :, None] / math.sqrt(math.sqrt(K.size(2)))
            X2 = torch.matmul(Q2, torch.matmul(torch.transpose(K2, -2, -1), V))
            
            if self.type_blend == 8:
                X = self.lowrank_ratio * X + self.lowrank_ratio2 * X2
            elif self.type_blend == 9:
                X = self.lowrank_ratio * X + self.lowrank_ratio2 * X2
            elif self.type_blend == 13:
                X = self.lowrank_ratio * X + self.lowrank_ratio2 * X2
            elif self.type_blend == 14:
                X = self.lowrank_ratio * X + self.lowrank_ratio2 * X2
            elif self.type_blend == 15:
                X = X + self.lowrank_ratio * X2
            elif self.type_blend == 16:
                X = self.lowrank_ratio * X + self.lowrank_ratio2 * X2
            elif self.type_blend == 17:
                X = X + self.lowrank_ratio * X2
            elif self.type_blend == 19:
                X = self.lowrank_ratio * X + self.lowrank_ratio2 * X2
            else:
                X = 0.5 * X + 0.5 * X2

        return X
    
    def _forward_sparse(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])
        
        #### Computing masked softmax(sparse) ####
        bsz, n_head, qlen, klen = dot.shape
        
        sparse_mask = torch.ones(qlen, klen).to(dot)
        sparse_mask = torch.tril(sparse_mask, diagonal=-self.diag_size) + torch.triu(sparse_mask, diagonal=self.diag_size)
        sparse_mask = sparse_mask.to(torch.bool)
        
        dot.masked_fill_(sparse_mask[None, None, :, :], -float('inf'))
        
        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X
