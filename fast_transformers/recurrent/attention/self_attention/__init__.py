#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Autoregressive implementations for self attention as a recurrent module.

The attention implementations in this module expect one input for query, one
for key and one for value and attend to all the keys and values seen so far. No
masking is necessary as an implicit lower triangular attention mask is assumed
in all cases.

Example
-------

    import torch

    from fast_transformers.recurrent.attention import \
        RecurrentAttentionLayer, RecurrentFullAttention

    att = RecurrentAttentionLayer(RecurrentFullAttention(), 16, 4)
    state = None
    x = torch.rand(8, 16)
    for i in range(10):
        x, state = att(x, x, x, state=state)
"""

from .attention_layer import RecurrentAttentionLayer
from .full_attention import RecurrentFullAttention
from .linear_attention import RecurrentLinearAttention
from .momentum_attention import RecurrentMomentumAttention
from .adam_attention import RecurrentAdamAttention
from .adamr_attention import RecurrentAdamRAttention
from .normgrad_attention import RecurrentNormgradAttention
from .frmomentum_attention import RecurrentFRMomentumAttention
from .nesterov_attention import RecurrentNesterovAttention
from .lsmomentum_attention import RecurrentLSMomentumAttention
from .adamax_attention import RecurrentAdamaxAttention
from .adamclip_attention import RecurrentAdamclipAttention
from .fradamax_attention import RecurrentFRAdamaxAttention
from .fradamax_attention_v2 import RecurrentFRAdamaxAttention_v2
