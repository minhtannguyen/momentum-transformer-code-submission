"""Implement the causally masked linear attention as a recurrent model."""

import torch
from torch.nn import Module

from ....attention_registry import RecurrentAttentionRegistry, Optional, Float, Int, \
    Callable, EventDispatcherInstance
from ....events import EventDispatcher
from ....feature_maps import elu_feature_map
from ..._utils import check_state


class RecurrentNesterovAttention(Module):
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
    def __init__(self, query_dimensions, stepsize, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(RecurrentNesterovAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
        # for momentum transformer
        self.stepsize = stepsize

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
            time_indx = 0
        else:
            Si, Zi, Pi, time_indx = state

        # Ensure the batch size did not change
        if len(Si) != N:
            raise ValueError("The batch size changed during iteration")

        # Update the internal state
        #
        # NOTE: The if clause is added due to GitHub PR #10. Simply using the
        # following two lines does not perform the operation in place which
        # means it is slower for inference.
        if K.grad_fn is not None or value.grad_fn is not None:
            time_indx = time_indx + 1
            mu = (time_indx - 1.0)/(time_indx + 2.0)
            Zi = Zi + K
            Pinext = Si + self.stepsize * torch.einsum("nhd,nhm->nhdm", K, value)
            Si = Pinext + mu * (Pinext - Pi)
            
        else:
            time_indx += 1
            mu = (time_indx - 1.0)/(time_indx + 2.0)
            Zi += K
            Pinext = Si + self.stepsize * torch.einsum("nhd,nhm->nhdm", K, value)
            Si = Pinext + mu * (Pinext - Pi)

        # Compute the output
        Z = 1. / (torch.einsum("nhd,nhd->nh", Q, Zi) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Z)

        return V, [Si, Zi, Pinext, time_indx]


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
    "nesterov-linear", RecurrentNesterovAttention,
    [
        ("query_dimensions", Int),
        ("stepsize", Float),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
