# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import math
import re
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from itertools import combinations_with_replacement
from typing import Any

import numpy as np
import scipy.linalg as slin
from scipy.special import erf

from ....common import PaddingType, find_dominant_type
from ...utils import NestedFloatOrIntOrBoolList, is_tuple_int
from .utils import (
    CacheType,
    calc_prob_matrix,
    calculate_binary_class_weight,
    calculate_cross_entropy_class_weights,
    determine_dtype,
    dtype_map,
    find_optimal_sigmas,
    get_submatrices1d,
    get_submatrices2d,
    log_sigmoid,
    log_softmax,
    write_into_cache,
)

AxisType = None | int | Sequence[int]

__all__ = [
    "partial",
    "exp",
    "sqrt",
    "sin",
    "cos",
    "abs",
    "sign",
    "log",
    "robust_power",
    "robust_sqrt",
    "robust_log",
    "stable_reciprocal",
    "relu",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "softplus",
    "cast",
    "gelu",
    "softmax",
    "reduce_mean",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "variance",
    "conv1d",
    "conv1d_bias",
    "conv2d",
    "conv2d_bias",
    "max_pool1d",
    "max_pool2d",
    "scaled_dot_product_attention",
    "cross_entropy",
    "cross_entropy_with_logits",
    "cross_entropy_with_log_probs",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "quantile_loss",
    "hinge_loss",
    "quad_hinge_loss",
    "kl_divergence",
    "absolute_error",
    "primitive_accuracy",
    "auc_core",
    "transposed_diag",
    "broadcast_to",
    "ones_with_zero_diag",
    "eye",
    "squeeze",
    "to_tensor",
    "tensor_to_list",
    "primitive_embedding",
    "where",
    "concat",
    "arange",
    "flatten",
    "stop_gradient",
    "shape",
    "size",
    "norm_modifier",
    "distance_matrix",
    "polynomial_features",
    "tsne_p_joint",
    "cholesky",
    "gpr_alpha",
    "eigvalsh",
    "gpr_v_outer",
    "isnan",
    "nan_to_num",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "logical_not",
    "logical_or",
    "logical_and",
    "logical_xor",
    "matrix_multiplication",
    "multiplication",
    "divide",
    "floor_divide",
    "shift_left",
    "shift_right",
    "minus",
    "add",
    "subtract",
    "power",
    "squared_error",
    "transpose",
    "square",
    "buffer",
    "permute_tensor",
    "reshape",
    "item",
    "indexer",
    "primitive_slice",
    "swapaxes",
    "sequence_slice",
    "union",
    "length",
    "cartesian_diff",
    "to_tuple",
    "to_list",
    "padding_converter_1d",
    "padding_converter_2d",
    "stride_converter",
    "tuple_converter",
    "make_array",
    "reduce_argmin",
    "reduce_argmax",
    "unique",
    "trapezoid",
    "pad",
    "split",
    "randn",
    "atleast_1d",
    "minimum",
    "maximum",
    "dtype",
    "zeros_like",
    "avg_pool2d",
]


# TODO: Type annotations of numpy functions are written as np.ndarray[Any, Any] for now,
# However, it can be annotated more precisely in some functions
# (example: np.ndarray[tuple[int, int, *tuple[int, ...]], np.dtype[np.float32]]).
# Above example annotates given arg will have at least two dimensions and
# it has np.float32 dtype. This kind of annotations can be added in the future.


# Ops
def exp(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    output = np.exp(input)
    return output


def sqrt(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    output = np.sqrt(input)
    return output


def sin(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    output = np.sin(input)
    return output


def cos(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    output = np.cos(input)
    return output


def abs(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.abs(input)


def sign(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.sign(input)


def log(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.log(input)


def unique(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.unique(input)


def trapezoid(
    y: np.ndarray[Any, Any], x: np.ndarray[Any, Any] | None = None
) -> np.float64 | np.ndarray[Any, Any]:
    return np.trapezoid(y, x)


def robust_power(
    base: np.ndarray[Any, Any],
    exponent: np.ndarray[Any, Any],
    threshold: np.ndarray[tuple[()], Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    # Broadcasting threshold for shape calculations
    threshold = np.resize(threshold, exponent.shape)
    # cond = (base < threshold) & (exponent < 1.0)
    cond = (exponent < (1 - np.log(threshold) / np.log(np.abs(base)))) & (
        np.abs(base) < threshold
    )
    if cache is not None:
        cache["cond"] = cond
    dummy_result = np.zeros_like(cond, dtype=base.dtype)
    if np.any(cond):
        dummy_result[cond] = ((1 / threshold) * np.abs(base))[cond]
    dummy_result[~cond] = (np.abs(base) ** exponent)[~cond]
    return dummy_result


# NOTE: We wrote the stabilized log in order to avoid
# undefined points (log(0) = -inf in this case),
# further testing should be done about performance
def robust_sqrt(
    input: np.ndarray[Any, Any],
    cutoff: np.ndarray[tuple[()], Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    input = np.abs(input)
    inds = input < cutoff
    output = np.zeros_like(input)
    output[~inds] = np.sqrt(input[~inds])
    output[inds] = input[inds] * np.reciprocal(np.sqrt(cutoff))
    return output


def robust_log(
    input: np.ndarray[Any, Any],
    cutoff: np.ndarray[tuple[()], Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    input = np.abs(input)
    inds = input < cutoff
    y_c = np.log(cutoff)
    output = np.zeros_like(input)
    output[~inds] = np.log(input[~inds])
    # Handle the values smaller than cutoff.
    output[inds] = y_c + (input[inds] / cutoff) - 1.0
    return output


# NOTE: We wrote stable reciprocal in order to handle
# undefined points (f(0) = inf in this case),
# futher testing should be done.
def stable_reciprocal(
    input: np.ndarray[Any, Any],
    cutoff: np.ndarray[tuple[()], Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    inds = np.abs(input) < cutoff
    y_c = np.reciprocal(cutoff)
    output = np.zeros_like(input)
    output[~inds] = np.reciprocal(input[~inds])
    # Handle the values smaller than cutoff.
    output[inds] = (
        np.sign(input[inds]) + (1 - np.sign(np.abs(input[inds])))
    ) * 2 * y_c + (-input[inds] / np.square(cutoff))
    return output


# Non linearity funcs
def relu(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.maximum(np.array(0.0, dtype=input.dtype), input)


def leaky_relu(
    input: np.ndarray[Any, Any],
    slope: float | np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.maximum(np.array(0.0, dtype=input.dtype), input) + slope * np.minimum(
        np.array(0.0, dtype=input.dtype), input
    )


def tanh(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.tanh(input)


def sigmoid(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    # For numerical stability implement sigmoid with respect to the
    # sign of input.
    mask = input >= 0
    sig = np.zeros_like(input)
    sig[mask] = 1.0 / (1.0 + np.exp(-input[mask]))
    sig[~mask] = np.exp(input[~mask]) / (1.0 + np.exp(input[~mask]))
    return sig


def softplus(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    # See: https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    return np.log1p(np.exp(-np.abs(input))) + np.maximum(input, 0.0)


def gelu(
    input: np.ndarray[Any, Any], approximate: bool, cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    if approximate:
        return (
            0.5
            * input
            * (1 + np.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * input**3)))
        )
    else:
        return input * (1 + erf(input / np.sqrt(2))) / 2


def softmax(
    input: np.ndarray[Any, Any], *, axis: int = -1, cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    write_into_cache(cache, "axis", axis)
    input_tensor = input - np.max(input, axis=axis, keepdims=True)
    e = np.exp(input_tensor)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s


# Reduction ops
def reduce_mean(
    input: np.ndarray[Any, Any],
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.mean(input, axis=axis, keepdims=keepdim)


def reduce_sum(
    input: np.ndarray[Any, Any],
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.sum(input, axis=axis, keepdims=keepdim)


def reduce_max(
    input: np.ndarray[Any, Any],
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.max(input, axis=axis, keepdims=keepdim)


def reduce_argmax(
    input: np.ndarray[Any, Any],
    *,
    axis: int | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.argmax(input, axis=axis, keepdims=keepdim)


def reduce_min(
    input: np.ndarray[Any, Any],
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.min(input, axis=axis, keepdims=keepdim)


def reduce_argmin(
    input: np.ndarray[Any, Any],
    *,
    axis: int | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.argmin(input, axis=axis, keepdims=keepdim)


def reduce_prod(
    input: np.ndarray[Any, Any],
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.prod(input, axis=axis, keepdims=keepdim)


def variance(
    input: np.ndarray[Any, Any],
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    correction: float = 0.0,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.var(input, axis=axis, ddof=correction, keepdims=keepdim)


# NN ops
def conv1d(
    input: np.ndarray[Any, Any],
    weight: np.ndarray[Any, Any],
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if dilation != 1:
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            f"Currently, the Numpy backend for conv2d only supports a dilation of 1."
        )
    n, c, w = input.shape
    *_, w_k = weight.shape
    out_w = (w - w_k + sum(padding)) // stride + 1
    submatrices = get_submatrices1d(input, (n, c, out_w), w_k, padding, stride)
    return np.einsum("niwl,oil->now", submatrices, weight)


def conv1d_bias(
    input: np.ndarray[Any, Any],
    weight: np.ndarray[Any, Any],
    bias: np.ndarray[Any, Any],
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return (
        conv1d(
            input=input,
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            cache=cache,
        )
        + bias
    )


def conv2d(
    input: np.ndarray[Any, Any],
    weight: np.ndarray[Any, Any],
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if dilation != (1, 1):
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            f"Numpy backend for conv2d only supports a dilation of (1, 1)."
        )

    _padding: tuple[tuple[int, int], tuple[int, int]]
    if is_tuple_int(padding):
        _padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        _padding = padding  # type: ignore

    n, c, h, w = input.shape
    _, _, h_k, w_k = weight.shape
    out_h = (h - h_k + sum(_padding[0])) // stride[0] + 1
    out_w = (w - w_k + sum(_padding[1])) // stride[1] + 1
    submatrices = get_submatrices2d(
        input, (n, c, out_h, out_w), h_k, w_k, _padding, stride[0]
    )
    return np.einsum("nihwkl,oikl->nohw", submatrices, weight)


def conv2d_bias(
    input: np.ndarray[Any, Any],
    weight: np.ndarray[Any, Any],
    bias: np.ndarray[Any, Any],
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return (
        conv2d(
            input=input,
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            cache=cache,
        )
        + bias
    )


def max_pool1d(
    input: np.ndarray[Any, Any],
    kernel_size: int,
    stride: int,
    *,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if dilation != 1:
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            f"Currently, the Numpy backend for Maxpool1d only supports a dilation of 1."
        )

    n, c, w = input.shape
    out_w = (w - kernel_size + sum(padding)) // stride + 1
    submatrices = get_submatrices1d(input, (n, c, out_w), kernel_size, padding, stride)
    return np.nanmax(submatrices, axis=3)


def max_pool2d(
    input: np.ndarray[Any, Any],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    *,
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    """Implements torch.nn.functional.max_pool2d in Numpy"""

    if dilation != (1, 1):
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            "Numpy backend for Maxpool2d only supports a dilation of (1, 1)."
        )

    normalized_padding: tuple[tuple[int, int], tuple[int, int]]
    if is_tuple_int(padding):
        normalized_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        normalized_padding = padding  # type: ignore

    n, c, h, w = input.shape
    out_h = (h - kernel_size[0] + sum(normalized_padding[0])) // stride[0] + 1
    out_w = (w - kernel_size[1] + sum(normalized_padding[1])) // stride[1] + 1
    submatrices = get_submatrices2d(
        input,
        (n, c, out_h, out_w),
        kernel_size[0],
        kernel_size[1],
        normalized_padding,
        stride[0],
    )
    return np.nanmax(submatrices, axis=(4, 5))


def avg_pool2d(
    input: np.ndarray[Any, Any],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    *,
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    """Implements torch.nn.functional.max_pool2d in Numpy"""

    if dilation != (1, 1):
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            "Numpy backend for Maxpool2d only supports a dilation of (1, 1)."
        )

    normalized_padding: tuple[tuple[int, int], tuple[int, int]]
    if is_tuple_int(padding):
        normalized_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        normalized_padding = padding  # type: ignore

    n, c, h, w = input.shape
    out_h = (h - kernel_size[0] + sum(normalized_padding[0])) // stride[0] + 1
    out_w = (w - kernel_size[1] + sum(normalized_padding[1])) // stride[1] + 1
    submatrices = get_submatrices2d(
        input,
        (n, c, out_h, out_w),
        kernel_size[0],
        kernel_size[1],
        normalized_padding,
        stride[0],
    )
    return np.nanmean(submatrices, axis=(4, 5))


def scaled_dot_product_attention(
    query: np.ndarray[Any, Any],
    key: np.ndarray[Any, Any],
    value: np.ndarray[Any, Any],
    attn_mask: np.ndarray[Any, Any] | None = None,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: bool | None = None,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if dropout_p != 0.0:
        raise RuntimeError(
            "Currently Numpy scaled_dot_product_attention only support dropout_p 0"
        )

    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / np.sqrt(query.shape[-1]) if scale is None else scale
    write_into_cache(cache, "scale_factor", scale_factor)
    attn_bias = np.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = np.tril(np.ones((L, S), dtype=bool))
        attn_bias = np.where(temp_mask, attn_bias, float("-inf"))
        attn_bias.astype(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == bool:
            attn_bias = np.where(attn_mask, attn_bias, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ np.swapaxes(key, -2, -1) * scale_factor
    write_into_cache(cache, "attn_weight_soft_out", attn_weight)
    attn_weight += attn_bias
    attn_weight = softmax(attn_weight, axis=-1)
    write_into_cache(cache, "attn_weight_soft_out", attn_weight)
    return attn_weight @ value


# Loss funcs
def cross_entropy(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weights: list[float] | bool,
    cutoff: np.ndarray[Any, Any],
    *,
    categorical: bool = True,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    write_into_cache(cache, "weights", _weights)
    log: partial[np.ndarray[Any, Any]] | Callable[..., np.ndarray[Any, Any]] = (
        partial(robust_log, cutoff=cutoff, cache=None) if robust else np.log
    )
    if categorical:
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )

        return (
            -log(np.take_along_axis(input, target[:, None], axis=1)[:, 0])
            * _weights[target]
        )
    return -np.sum(target * log(input) * _weights, axis=1)


def cross_entropy_with_logits(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weights: list[float] | bool,
    cutoff: np.ndarray[Any, Any],
    *,
    categorical: bool = True,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    log: partial[np.ndarray[Any, Any]] | Callable[..., np.ndarray[Any, Any]] = (
        partial(robust_log, cutoff=cutoff, cache=None) if robust else np.log
    )
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    write_into_cache(cache, "weights", _weights)
    if categorical:
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )

        logits_max = np.max(input, axis=1, keepdims=True)
        input_norm = input - logits_max
        label_logits = np.take_along_axis(input_norm, target[:, None], axis=1)[:, 0]
        log_normalizers = log(np.sum(np.exp(input_norm), axis=1))
        return (log_normalizers - label_logits) * _weights[target]

    return -np.sum(target * log_softmax(input, log, robust, axis=1) * _weights, axis=1)


def cross_entropy_with_log_probs(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weights: list[float] | bool,
    *,
    categorical: bool = True,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    write_into_cache(cache, "weights", _weights)
    if categorical:
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError(
                f"Crossentropy got unexpected type for target '{target.dtype}'"
            )

        return (
            -np.take_along_axis(input, target[:, None], axis=1)[:, 0] * _weights[target]
        )
    return -np.sum(target * input * _weights, axis=1)


def binary_cross_entropy(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    cutoff: np.ndarray[Any, Any],
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    log: partial[np.ndarray[Any, Any]] | Callable[..., np.ndarray[Any, Any]] = (
        partial(robust_log, cutoff=cutoff, cache=None) if robust else np.log
    )
    if isinstance(pos_weight, bool) and pos_weight:
        pos_weight = float(calculate_binary_class_weight(target))
    write_into_cache(cache, "pos_weight", pos_weight)
    return -pos_weight * target * log(input) - (1 - target) * log(1 - input)


def binary_cross_entropy_with_logits(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    cutoff: np.ndarray[Any, Any],
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    log: partial[np.ndarray[Any, Any]] | Callable[..., np.ndarray[Any, Any]] = (
        partial(robust_log, cutoff=cutoff, cache=None) if robust else np.log
    )

    if isinstance(pos_weight, bool):
        _pos_weight = (
            calculate_binary_class_weight(sigmoid(target)) if pos_weight else 1.0
        )
    else:
        _pos_weight = pos_weight

    if _pos_weight != 1.0:
        write_into_cache(cache, "pos_weight", _pos_weight)
        log_weight = (_pos_weight - 1) * (target) + 1
        loss = (1 - target) * input - (log_weight * log_sigmoid(input, log, robust))
    else:
        loss = (1 - target) * input - log_sigmoid(input, log, robust)

    return loss


def quantile_loss(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    quantile: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    error = target - input
    return np.maximum(quantile * error, (quantile - 1) * error)


def hinge_loss(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    base_hinge = 1.0 - target * input
    write_into_cache(cache, "base_hinge", base_hinge)
    return np.maximum(0.0, base_hinge)


def quad_hinge_loss(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return hinge_loss(input, target) ** 2


def kl_divergence(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    cutoff: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    log_input1 = robust_log(input, cutoff)
    log_input2 = robust_log(target, cutoff)
    partial_result = log_input2 - log_input1
    write_into_cache(cache, "partial_result", partial_result)
    return target * partial_result


def absolute_error(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    diff = input - target
    write_into_cache(cache, "diff", diff)
    return np.abs(diff)


def primitive_accuracy(
    input1: np.ndarray[Any, Any],
    input2: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    prediction = np.argmax(input1, axis=1).reshape(input1.shape[0], 1)
    return np.mean(prediction == input2)


def auc_core(
    input: np.ndarray[Any, Any],
    label: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if input.ndim > 1:
        raise ValueError(f"Input should be 1D array, but given '{input.ndim}D'")
    if label.ndim > 1:
        raise ValueError(f"Label should be 1D array, but given '{label.ndim}D'")

    n_positive = (label == 1).sum()
    n_negative = len(label) - n_positive
    sorted_input = np.sort(input)
    tprs = []
    fprs = []

    # TODO: This is very inefficient, improve it.
    for threshold in np.flip(np.unique(sorted_input)):
        input_c = input.copy()
        input_c[input_c >= threshold] = 1
        input_c[input_c < threshold] = 0

        true_positives = np.sum((input_c == 1) & (label == 1))
        false_positives = np.sum((input_c == 1) & (label == 0))

        fprs.append(false_positives / n_negative)
        tprs.append(true_positives / n_positive)

    tprs = np.stack(tprs)
    fprs = np.stack(fprs)

    return np.stack([tprs, fprs])


def transposed_diag(
    input: np.ndarray[Any, Any], *, cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.diag(input)[:, np.newaxis]


def broadcast_to(
    input: np.ndarray[Any, Any], shape: tuple[int, ...], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.broadcast_to(input, shape)


def squeeze(
    input: np.ndarray[Any, Any], *, cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.squeeze(input)


def to_tensor(
    *input: NestedFloatOrIntOrBoolList,
    dtype: np.dtype[Any] | None = None,
    default_dtype: str,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    dtype_str = default_dtype if dtype is None else dtype_map.inverse[dtype]

    dominant_type = find_dominant_type(input)
    _dtype = dominant_type.__name__

    if _dtype != "bool":
        _dtype += str(re.findall(r"\d+", dtype_str)[-1])

    return np.array(input[0], dtype=_dtype)


def make_array(
    input: int | float | np.ndarray[Any, Any],
    *,
    dtype: str | None = None,
    default_dtype: str,
) -> np.ndarray[Any, Any]:
    if dtype is None:
        dtype = default_dtype

    dtype = determine_dtype(
        input,
        None,
        default_dtype,
        precision=int(re.findall(r"\d+", dtype)[-1]),
    )

    return np.array(input, dtype=dtype_map[dtype])


def eye(
    N: int,
    M: int | None,
    *,
    dtype: np.dtype[Any] | None = None,
    default_dtype: str,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    dtype = dtype_map[default_dtype] if dtype is None else dtype

    return np.eye(N, M, dtype=dtype)


def ones_with_zero_diag(
    N: int,
    M: int | None,
    *,
    dtype: np.dtype[Any] | None = None,
    default_dtype: str,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    dtype = dtype_map[default_dtype] if dtype is None else dtype

    output = (
        np.ones((N, M), dtype=dtype) - np.eye(N, M, dtype=dtype)
        if M is not None
        else np.ones(N, dtype=dtype) - np.eye(N, dtype=dtype)
    )

    return output


def arange(
    start: int | float,
    stop: int | float,
    step: int | float,
    *,
    dtype: np.dtype[Any] | None = None,
    default_dtype: str,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    _dtype = default_dtype if dtype is None else dtype_map.inverse[dtype]

    if len([item for item in [start, stop, step] if isinstance(item, float)]) == 0:
        _dtype = _dtype.replace("bfloat", "int").replace("float", "int")

    return np.arange(start, stop, step, dtype=_dtype)


def tensor_to_list(input: np.ndarray[Any, Any], cache: CacheType | None = None) -> Any:
    return input.tolist()


def primitive_embedding(
    input: np.ndarray[Any, Any],
    weight: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return weight[input]


def minimum(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.minimum(left, right)


def maximum(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.maximum(left, right)


def where(
    cond: np.ndarray[Any, Any],
    input1: np.ndarray[Any, Any],
    input2: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.where(cond, input1, input2)


def concat(
    input: list[np.ndarray[Any, Any]],
    axis: int | None = 0,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.concatenate(input, axis=axis)


def flatten(
    input: np.ndarray[Any, Any],
    *,
    start_dim: int = 0,
    end_dim: int = -1,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    """Flattens a Numpy array akin to torch.flatten"""
    if end_dim == -1 or end_dim == len(input.shape):
        end_dim = len(input.shape) + 1
    end_dim = (
        int(end_dim == -1 or end_dim == len(input.shape)) * (len(input.shape) + 1)
        + int(end_dim != -1 and end_dim != len(input.shape)) * end_dim
    )
    prod = np.prod(input.shape[start_dim : end_dim + 1]).astype(int)
    shape = input.shape[:start_dim] + (prod,) + input.shape[end_dim + 1 :]
    return np.reshape(input, shape)


def stop_gradient(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return input


def shape(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> tuple[int, ...]:
    return input.shape


def size(
    input: np.ndarray[Any, Any],
    dim: int | tuple[int, ...] | None,
    cache: CacheType | None = None,
) -> int | tuple[int, ...]:
    if dim is None:
        return input.size
    if isinstance(dim, int):
        return input.shape[dim]
    else:
        return tuple(input.shape[idx] for idx in dim)


def norm_modifier(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    inner_term = ((input - 1.0) % 8) / 8 - 0.5
    write_into_cache(cache, "inner_term", inner_term)
    return 4.0 * (1.0 - 2.0 * np.abs(inner_term)) + 1.0


def distance_matrix(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    norm: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    diffs = left[:, None, :] - right[None, :, :]
    write_into_cache(cache, "diffs", diffs)
    abs_diffs = np.abs(diffs)
    write_into_cache(cache, "abs_diffs", abs_diffs)
    powered_abs_diffs = abs_diffs**norm
    write_into_cache(cache, "powered_abs_diffs", powered_abs_diffs)
    return np.sum(powered_abs_diffs, axis=2)


def polynomial_features(
    input: np.ndarray[tuple[int, int], Any],
    *,
    degree: int = 2,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    samples, dims = input.shape
    identity = np.eye(dims + 1, dims + 1, dtype=int)
    data = np.hstack((np.ones((samples, 1), dtype=input.dtype), input))
    write_into_cache(cache, "data", data)
    powers: Iterator[int] = map(sum, combinations_with_replacement(identity, degree))
    # Skip first element of powers. This is the bias term.
    next(powers)
    write_into_cache(
        cache,
        "powers",
        copy.deepcopy(powers),  # type: ignore
    )  # Copy iterator before it is iterated.
    return np.hstack([(data**p).prod(1)[:, np.newaxis] for p in powers])


def tsne_p_joint(
    squared_distances: np.ndarray[Any, Any],
    target_perplexity: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    """Given a data matrix X, gives joint probabilities matrix.
    Parameters
    ----------
    squared_distances : np.ndarray[Any, Any]
        Square of distance matrix of Input data.
    target_perplexity : float
        Desired perplexity value.
    Returns
    -------
    np.ndarray
        Matrix with entries p_ij: joint probabilities.
    """
    # NOTE: There is no gradient function for this primitive model because this
    # function is performed only once throughout training!!!

    # Get the negative euclidian distances matrix for our data
    negative_dist_sq = (-1) * squared_distances
    # Find optimal sigma for each row of this distances matrix
    # TODO: fix types
    sigmas = find_optimal_sigmas(negative_dist_sq, target_perplexity, threshold)  # type: ignore
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(negative_dist_sq, sigmas)
    # Go from conditional to joint probabilities matrix in a symmetrical manner.
    return (p_conditional + p_conditional.T) / (2.0 * p_conditional.shape[0])


def cholesky(
    input1: np.ndarray[Any, Any], *, cache: CacheType | None = None
) -> np.ndarray[Any, Any] | None:
    try:
        return np.linalg.cholesky(input1)
    except np.linalg.LinAlgError as e:
        logging.info(str(e))
    return None


def gpr_alpha(
    label_mu_diff: np.ndarray[Any, Any],
    L: np.ndarray[Any, Any],
    K_term: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if L is not None:
        alpha = slin.solve_triangular(
            L.T,
            slin.solve_triangular(L, label_mu_diff, lower=True),
            lower=False,
        )
    else:
        alpha = np.linalg.solve(K_term, label_mu_diff)
    return alpha


def eigvalsh(
    K_term: np.ndarray[Any, Any],
    L: np.ndarray[Any, Any],
    threshold: float,
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if L is not None:
        return np.diag(L)
    else:
        return np.clip(np.linalg.eigvalsh(K_term), threshold, None)


def gpr_v_outer(
    K: np.ndarray[Any, Any],
    K_term: np.ndarray[Any, Any],
    L: np.ndarray[Any, Any],
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if L is not None:
        v = slin.solve_triangular(L, K, lower=True)
        v_outer = v.T @ v
    else:
        v_outer = K.T @ np.linalg.lstsq(K_term, K)[0]
    return v_outer


def isnan(
    input: np.ndarray[Any, Any], *, cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.isnan(input)


def nan_to_num(
    input: np.ndarray[Any, Any],
    nan: float,
    posinf: float | None,
    neginf: float | None,
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)


def cast(input: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    return input.astype(dtype)


def dtype(input: np.ndarray[Any, Any]) -> np.dtype[Any]:
    return input.dtype.type


def logical_xor(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.logical_xor(left, right)


def split(
    input: np.ndarray[Any, Any],
    split_size: int | list[int],
    axis: int = 0,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.stack(np.split(input, split_size, axis=axis))


def pad(
    input: np.ndarray[Any, Any],
    pad_width: tuple[tuple[int, int], ...],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return np.pad(input, pad_width)


def randn(
    shape: tuple[int, ...],
    key: int,
    *,
    dtype: str | None = None,
    default_dtype: str,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    np.random.seed(key)
    if dtype is None:
        dtype = default_dtype
    return np.random.randn(*shape).astype(dtype)


def zeros_like(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.zeros_like(input)


def atleast_1d(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return np.atleast_1d(input)


def greater(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left > right


def greater_equal(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left >= right


def less(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left < right


def less_equal(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left <= right


def equal(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left == right


def not_equal(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left != right


def logical_not(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return ~input


def logical_or(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left | right


def logical_and(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left & right


def matrix_multiplication(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left @ right


def multiplication(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left * right


def divide(
    numerator: np.ndarray[Any, Any],
    denominator: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return numerator / denominator


def floor_divide(
    numerator: np.ndarray[Any, Any],
    denominator: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return numerator // denominator


def shift_left(
    input: np.ndarray[Any, Any],
    shift: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return input << shift


def shift_right(
    input: np.ndarray[Any, Any],
    shift: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return input >> shift


def minus(input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return -input


def add(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left + right


def subtract(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left - right


def power(
    base: np.ndarray[Any, Any],
    exponent: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return base**exponent


def squared_error(
    input: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return (input - target) ** 2


# def transpose(input: np.ndarray[Any, Any], cache: CacheType|None = None) :
#     return input.T


def transpose(
    input: np.ndarray[Any, Any],
    axes: list[int] | tuple[int, ...] | None = None,
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    if not axes:
        return input.T
    return input.transpose(*axes)


def square(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return input * input


def buffer(
    input: np.ndarray[Any, Any], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return input


def permute_tensor(
    input: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return input[indices]


def reshape(
    input: np.ndarray[Any, Any], shape: tuple[int, ...], cache: CacheType | None = None
) -> np.ndarray[Any, Any]:
    return input.reshape(shape)


def item(input: np.ndarray[Any, Any]) -> int | float | bool:
    return input.item()  # type: ignore


def sequence_slice(
    input: list[int | float] | tuple[int | float, ...],
    start: int | None,
    stop: int | None,
    step: int | None,
    cache: CacheType | None = None,
) -> list[int | float] | tuple[int | float, ...]:
    return input[start:stop:step]


def union(
    *args: int | float | tuple[int | float, ...], cache: CacheType | None = None
) -> tuple[int | float, ...]:
    result: tuple[int | float, ...] = tuple()
    for arg in args:
        result += arg if isinstance(arg, tuple) else (arg,)

    return result


def to_tuple(
    *args: tuple[int | float | bool, ...], cache: CacheType | None = None
) -> tuple[Any, ...]:
    return tuple(args)


def to_list(
    *args: tuple[int | float | bool, ...], cache: CacheType | None = None
) -> list[Any]:
    return list(args)


def padding_converter_1d(
    input: PaddingType | int | tuple[int, int],
    kernel_size: int | tuple[int, int],
    cache: CacheType | None = None,
) -> tuple[int, int]:
    if isinstance(input, PaddingType):
        if input == PaddingType.VALID:
            output = (0, 0)
        elif isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            half = kernel_size // 2
            output = (half, half)
        else:
            raise RuntimeError("Kernel size must be 'tuple[int, int]' or 'int'!")

    elif isinstance(input, int):
        output = (input, input)

    else:
        if isinstance(input[0], Sequence) or isinstance(input[1], Sequence):
            raise RuntimeError(f"Given input '{input}' is not valid!")
        output = input

    return output


def padding_converter_2d(
    input: PaddingType
    | int
    | tuple[int, int]
    | tuple[tuple[int, int] | tuple[int, int]],
    kernel_size: int | tuple[int, int],
    cache: CacheType | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(input, PaddingType):
        if input == PaddingType.VALID:
            output = ((0, 0), (0, 0))
        elif isinstance(kernel_size, tuple):
            if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            output = (
                (kernel_size[0] // 2, kernel_size[1] // 2),
                (kernel_size[0] // 2, kernel_size[1] // 2),
            )
        else:
            if kernel_size % 2 == 0:
                raise RuntimeError(
                    "'same' padding is not supported when the kernel size is even!"
                )
            half = kernel_size // 2
            output = ((half, half), (half, half))
    elif isinstance(input, int):
        output = ((input, input), (input, input))
    else:
        _output: list[tuple[int, int]] = []
        for p in input:
            if isinstance(p, int):
                _output.append((p, p))
            elif len(p) == 2:
                _output.append(p)

        output = ((_output[0][0], _output[0][1]), (_output[1][0], _output[1][1]))
    return output


# TODO: Overload this function.
def indexer(
    input: np.ndarray[Any, Any]
    | list[int | float | bool]
    | tuple[int | float | bool, ...],
    index: int | slice | tuple[int | slice, ...],
    cache: CacheType | None = None,
) -> (
    np.ndarray[Any, Any]
    | list[int | float | bool]
    | tuple[int | float | bool, ...]
    | int
    | float
    | bool
):
    return input[index]  # type: ignore


def primitive_slice(
    start: int | None,
    stop: int | None,
    step: int | None,
    cache: CacheType | None = None,
) -> slice:
    return slice(start, stop, step)


def swapaxes(
    input: np.ndarray[Any, Any],
    axis1: int,
    axis2: int,
    *,
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return input.swapaxes(axis1, axis2)


def stride_converter(
    input: int | tuple[int, int] | None,
    kernel_size: int | tuple[int, int],
    cache: CacheType | None = None,
) -> int | tuple[int, int]:
    if input is None:
        return kernel_size
    else:
        return input


def tuple_converter(
    input: int | tuple[int, int], cache: CacheType | None = None
) -> tuple[int, int]:
    if isinstance(input, int):
        return (input, input)
    else:
        return input


def length(input: np.ndarray[Any, Any]) -> int:
    return len(input)


def cartesian_diff(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    cache: CacheType | None = None,
) -> np.ndarray[Any, Any]:
    return left[:, None, :] - right[None, :, :]


array_creation_funcs = [
    "arange",
    "randn",
    "to_tensor",
    "make_array",
    "eye",
    "ones_with_zero_diag",
]
primitive_func_dict = {key: fn for key, fn in globals().items() if callable(fn)}
