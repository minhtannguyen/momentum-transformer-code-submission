#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement causally masked linear attention."""

import torch
import torch.nn as nn
from torch.nn import Module, Dropout
from math import sqrt

from ..attention_registry import AttentionRegistry, Optional, Callable, Int, Float, \
    EventDispatcherInstance
from ..events import EventDispatcher, AttentionEvent
from ..causal_product import causal_dot_product
from ..feature_maps import tanh_feature_map, elu_feature_map, relu_feature_map, celu_feature_map, sigmoid_feature_map, \
leakyrelu_feature_map, softplus_feature_map

def causal_linear(Q, K, V):
    Q = Q.permute(0,2,1,3).contiguous()
    K = K.permute(0,2,1,3).contiguous()
    V = V.permute(0,2,1,3).contiguous()
    V_new = causal_dot_product(Q, K, V)
    return V_new.permute(0,2,1,3).contiguous()


class CausalRankKAttention(Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.

    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher="", softmax_temp=None, attention_dropout=0.1,
                  diag_size=0, blend=0.9, kernel_1='tanh', kernel_2='none'):

        super(CausalRankKAttention, self).__init__()

        # first kernel function
        if kernel_1 == 'tanh':
            self.feature_map1 = tanh_feature_map(query_dimensions)
        elif kernel_1 == 'elu':
            self.feature_map1 = elu_feature_map(query_dimensions)
        elif kernel_1 == 'relu':
            self.feature_map1 = relu_feature_map(query_dimensions)
        elif kernel_1 == 'celu':
            self.feature_map1 = celu_feature_map(query_dimensions)
        elif kernel_1 == 'sigmoid':
            self.feature_map1 = sigmoid_feature_map(query_dimensions)
        elif kernel_1 == 'leakyrelu':
            self.feature_map1 = leakyrelu_feature_map(query_dimensions)
        elif kernel_1 == 'softplus':
            self.feature_map1 = softplus_feature_map(query_dimensions)

        # second kernel function
        if kernel_2 == 'tanh':
            self.feature_map2 = tanh_feature_map(query_dimensions)
        elif kernel_2 == 'elu':
            self.feature_map2 = elu_feature_map(query_dimensions)
        elif kernel_2 == 'relu':
            self.feature_map2 = relu_feature_map(query_dimensions)
        elif kernel_2 == 'celu':
            self.feature_map2 = celu_feature_map(query_dimensions)
        elif kernel_2 == 'sigmoid':
            self.feature_map2 = sigmoid_feature_map(query_dimensions)
        elif kernel_2 == 'leakyrelu':
            self.feature_map2 = leakyrelu_feature_map(query_dimensions)
        elif kernel_2 == 'softplus':
            self.feature_map2 = softplus_feature_map(query_dimensions)
        elif kernel_2 == 'none':
            self.feature_map2 = None

        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
        # Sparse components
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.diag_size = diag_size
        self.blend = nn.Parameter(torch.Tensor([blend]))
        self.kernel_blend = nn.Parameter(torch.Tensor([0.5]))

    def _make_sizes_compatible(self, Q, K):
        """Either slice or pad K in case that the sizes do not match between Q
        and K."""
        N, L, H, E = Q.shape
        _, S, _, _ = K.shape
        if L == S:
            return Q, K

        if L < S:
            return Q, K[:, :L, :, :]

        if L > S:
            return Q, torch.cat([K, K.new_zeros(N, L-S, H, E)], dim=1)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):

        # Apply the key padding mask and make sure the attn_mask is a
        # lower triangular causal mask
        if not attn_mask.lower_triangular:
            raise RuntimeError(("CausalLinearAttention only supports full "
                                "lower triangular masks"))

        # Apply the feature map to the queries and keys
        self.feature_map1.new_feature_map()
        Q1 = self.feature_map1.forward_queries(queries)
        K1 = self.feature_map1.forward_keys(keys)
        K1 = K1 * key_lengths.float_matrix[:, :, None, None]
        # Ensure that Q and K have compatible sizes for the following
        # computations, namely L == S
        Q1, K1 = self._make_sizes_compatible(Q1, K1)
        # Compute the normalizers
        Z1 = 1/(torch.einsum("nlhi,nlhi->nlh", Q1, K1.cumsum(1)) + self.eps)
        # Compute the unnormalized result
        V1 = causal_linear(
            Q1,
            K1,
            values
        )

        # Do the same for a second feature map
        if self.feature_map2:
            self.feature_map2.new_feature_map()
            Q2 = self.feature_map2.forward_queries(queries)
            K2 = self.feature_map2.forward_keys(keys)
            K2 = K2 * key_lengths.float_matrix[:, :, None, None]
            Q2, K2 = self._make_sizes_compatible(Q2, K2)
            Z2 = 1/(torch.einsum("nlhi,nlhi->nlh", Q2, K2.cumsum(1)) + self.eps)
            V2 = causal_linear(
                Q2,
                K2,
                values
            )
            Z = torch.sum(torch.stack([self.kernel_blend * Z1, (1.0 - self.kernel_blend) * Z2]), dim=0)
            V1 *= Z[:, :, :, None]
            V2 *= Z[:, :, :, None]
            LV = torch.sum(torch.stack([self.kernel_blend * V1, (1.0 - self.kernel_blend) * V2]), dim=0)
        else:
            LV = V1 * Z1[:, :, :, None]
        
        ## Sparse Attention
        
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]

        #compute the attention and the weighted average
        A = torch.softmax(softmax_temp * QK, dim=-1)
        A = torch.triu(A.view(-1, L, S), diagonal=-self.diag_size)
        A = A.view(N, H, L, S)
        A = self.dropout(A)
        SV = torch.einsum("nhls,nshd->nlhd", A, values)
        
        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return self.blend * SV.contiguous() + (1. - self.blend) *LV

        # # Lists which will store intermediate values for each feature map
        # Qs = []
        # KVs = []
        # Zs = []
        # Vs = []
        
        # # Apply the feature map to the queries and keys
        # for f in self.feature_map:
        #     f.new_feature_map(queries.device)
        #     Q = f.forward_queries(queries)
        #     K = f.forward_keys(keys)

        #     # Apply the key padding mask and make sure the attn_mask is a
        #     # lower triangular causal m ask
        #     if not attn_mask.lower_triangular:
        #         raise RuntimeError(("CausalRankKAttention only supports full "
        #                             " lower triangular masks"))
        #     K = K * key_lengths.float_matrix[:, :, None, None]

        #     # Ensure that Q and K have compatible sizes for the following
        #     # computations, namely L == S
        #     Q, K = self._make_sizes_compatible(Q, K)

        #     # Compute the normalizer for this feature map
        #     Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)

        #     # Compute the unnormalized result
        #     V = causal_linear(
        #         Q,
        #         K,
        #         values
        #     )

        #     Zs.append(Z)
        #     Vs.append(V)

        # # Sum the normalizers
        # Z = torch.sum(torch.stack(Zs), dim=0)

        # # Normalize each V
        # Vs = [V * Z[:, :, :, None] for V in Vs]

        # # # Compute sum of normalized Vs
        # # V = torch.sum(torch.stack(Vs), dim=0)

        # return torch.sum(torch.stack(Vs), dim=0)

# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "causal-rank-k", CausalRankKAttention,
    [  
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        ("diag_size", Optional(Int)),
        ("blend", Optional(Float)),
        ("kernel_1", Optional(EventDispatcherInstance, "")),
        ("kernel_2", Optional(EventDispatcherInstance, ""))
    ]
)