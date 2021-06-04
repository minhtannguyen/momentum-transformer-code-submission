#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement causally masked linear attention."""

import torch
from torch.nn import Module

import numpy as np

from ..attention_registry import AttentionRegistry, Optional, Callable, Float, Int, \
    EventDispatcherInstance
from ..events import EventDispatcher
from ..causal_product import causal_dot_product
from ..feature_maps import elu_feature_map


def causal_linear(Q, K, V):
    Q = Q.permute(0,2,1,3).contiguous()
    K = K.permute(0,2,1,3).contiguous()
    V = V.permute(0,2,1,3).contiguous()
    V_new = causal_dot_product(Q, K, V)
    return V_new.permute(0,2,1,3).contiguous()


class CausalNCMomentumAttention_Py(Module):
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
    def __init__(self, query_dimensions, stepsize, delta, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(CausalNCMomentumAttention_Py, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
        # for momentum
        self.stepsize = stepsize
        self.delta = delta

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
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map()
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure the attn_mask is a
        # lower triangular causal mask
        if not attn_mask.lower_triangular:
            raise RuntimeError(("CausalNCMomentumAttention only supports full "
                                "lower triangular masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Ensure that Q and K have compatible sizes for the following
        # computations, namely L == S
        Q, K = self._make_sizes_compatible(Q, K)

        # TODO: Shall we divide the Q and K with a relatively large number to
        #       avoid numerical instabilities in computing the denominator?
        #       We used to divide each with the max norm of all q and k but
        #       that seems relatively costly for a simple normalization.

        # Compute the normalizers
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)
        KV = torch.einsum("nshd,nshm->nhmd", K, values)
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
        
        ########### Unit Test ############
                
        # Compute the unnormalized result
        Zc = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
        Vw = causal_linear(
            Q,
            K,
            values
        )
        Vc = Vw * Zc[:, :, :, None]
        
        import pdb; pdb.set_trace()
        
        
        ##################################

        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "causal-ncmomentum-py", CausalNCMomentumAttention_Py,
    [
        ("query_dimensions", Int),
        ("stepsize", Float),
        ("delta", Float),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
