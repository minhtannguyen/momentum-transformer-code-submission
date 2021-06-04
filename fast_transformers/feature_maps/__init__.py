#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implementations of feature maps to be used with linear attention and causal
linear attention."""

from .base import tanh_feature_map, elu_feature_map, relu_feature_map, \
celu_feature_map, sigmoid_feature_map, \
leakyrelu_feature_map, softplus_feature_map, ActivationFunctionFeatureMap

from .fourier_features import RandomFourierFeatures, Favor, \
    SmoothedRandomFourierFeatures, GeneralizedRandomFeatures
