#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the causally masked linear attention as a recurrent model."""

import torch
from torch.nn import Module

from ....attention_registry import RecurrentAttentionRegistry, Optional, Float, Int, \
    Callable, EventDispatcherInstance
from ....events import EventDispatcher
from ....feature_maps import elu_feature_map
from ..._utils import check_state


class RecurrentAdamclipAttention(Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    See fast_transformers.attention.linear_attention and
    fast_transformers.attention.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.

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
    def __init__(self, query_dimensions, mu, stepsize, beta, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(RecurrentAdamclipAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
        # for adam transformer
        self.mu = mu
        self.stepsize = stepsize
        self.beta = beta

    def forward(self, query, key, value, state=None, memory=None):
        # Normalize state/memory
        state = check_state(state, memory)

        # If this is a new sequence reinitialize the feature map
        if state is None:
            self.feature_map.new_feature_map()

        # Apply the feature map to the query and key
        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)

        # Extract some shapes
        N, H, D = Q.shape
        _, _, M = value.shape

        # Extract the memory or initialize it
        if state is None:
            Si = query.new_zeros((N, H, D, M))
            Zi = query.new_zeros((N, H, D))
            Pi = query.new_zeros((N, H, D, M))
            Mi = query.new_zeros((N, H, D, M))
        else:
            Si, Zi, Pi, Mi, _ = state

        # Ensure the batch size did not change
        if len(Si) != N:
            raise ValueError("The batch size changed during iteration")

        # Update the internal state
        #
        # NOTE: The if clause is added due to GitHub PR #10. Simply using the
        # following two lines does not perform the operation in place which
        # means it is slower for inference.
        if K.grad_fn is not None or value.grad_fn is not None:
            Zi = Zi + K
            Ui = torch.einsum("nhd,nhm->nhdm", K, value)
            Pi = self.mu * Pi - self.stepsize * torch.clamp(Ui, min=-2.0, max=2.0)
            Mi = self.beta * Mi + (1.0 - self.beta) * (Ui * Ui)
            Si = Si - Pi/torch.sqrt(Mi + 1e-16)
              
        else:
            Zi += K
            Ui = torch.einsum("nhd,nhm->nhdm", K, value)
            Pi *= self.mu
            Pi -= self.stepsize * torch.clamp(Ui, min=-2.0, max=2.0)
            Mi *= self.beta
            Mi += (1.0 - self.beta) * (Ui * Ui)
            Si -= Pi/torch.sqrt(Mi + 1e-16)
            
        # Compute the output
        Z = 1. / (torch.einsum("nhd,nhd->nh", Q, Zi) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Z)

        return V, [Si, Zi, Pi, Mi, Ui]


# Register the attention implementation so that it becomes available in our
# builders

# RecurrentAttentionRegistry.register(
#     "momentum-linear", RecurrentMomentumAttention,
#     [
#         ("query_dimensions", Int),
#         ("feature_map", Optional(Callable)),
#         ("event_dispatcher", Optional(EventDispatcherInstance, ""))
#     ]
# )

RecurrentAttentionRegistry.register(
    "adamclip-linear", RecurrentAdamclipAttention,
    [
        ("query_dimensions", Int),
        ("mu", Float),
        ("stepsize", Float),
        ("beta", Float),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
