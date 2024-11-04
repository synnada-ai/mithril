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
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import scipy.linalg as slin  # type: ignore[import-untyped]
from scipy.special import erf  # type: ignore[import-untyped]

from .... import core
from ....utils.type_utils import is_tuple_int
from ...utils import NestedFloatOrIntOrBoolList
from ..common_primitives import (
    add,
    buffer,
    cartesian_diff,
    common_primitive_func_dict,
    divide,
    equal,
    floor_divide,
    greater,
    greater_equal,
    item,
    length,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    matrix_multiplication,
    minus,
    multiplication,
    not_equal,
    padding_converter_1d,
    padding_converter_2d,
    permute_tensor,
    power,
    reshape,
    scalar_item,
    sequence_slice,
    shift_left,
    shift_right,
    square,
    squared_error,
    stride_converter,
    subtract,
    swapaxes,
    tensor_item,
    tensor_slice,
    to_list,
    to_tuple,
    transpose,
    tuple_converter,
    union,
)
from .utils import (
    CacheType,
    calc_prob_matrix,
    calculate_binary_class_weight,
    calculate_cross_entropy_class_weights,
    find_optimal_sigmas,
    get_submatrices1d,
    get_submatrices2d,
    get_type,
    handle_data_dtype,
    handle_data_precision,
    log_sigmoid,
    log_softmax,
    make_array,
    write_into_cache,
)

np._set_promotion_state("legacy")

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
    "tensor_slice",
    "buffer",
    "permute_tensor",
    "reshape",
    "item",
    "scalar_item",
    "tensor_item",
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
    "common_primitive_func_dict",
    "reduce_argmin",
    "reduce_argmax",
    "unique",
    "trapezoid",
]


# Ops
def exp(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    output = np.exp(input)
    return output


def sqrt(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    output = np.sqrt(input)
    return output


def sin(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    output = np.sin(input)
    return output


def cos(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    output = np.cos(input)
    return output


def abs(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return np.abs(input)


def sign(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return np.sign(input)


def log(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return np.log(input)


def unique(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return np.unique(input)


def trapezoid(y: np.ndarray, x: np.ndarray | None = None) -> np.float64 | np.ndarray:
    return np.trapezoid(y, x)


def robust_power(
    base: np.ndarray,
    exponent: np.ndarray,
    threshold: np.ndarray,
    cache: CacheType | None = None,
) -> np.ndarray:
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
    input: np.ndarray, cutoff: np.ndarray, cache: CacheType | None = None
) -> np.ndarray:
    input = np.abs(input)
    inds = input < cutoff
    output = np.zeros_like(input)
    output[~inds] = np.sqrt(input[~inds])
    output[inds] = input[inds] * np.reciprocal(np.sqrt(cutoff))
    return output


def robust_log(
    input: np.ndarray, cutoff: np.ndarray, cache: CacheType | None = None
) -> np.ndarray:
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
    input: np.ndarray, cutoff: np.ndarray, cache: CacheType | None = None
) -> np.ndarray:
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
def relu(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return np.maximum(np.array(0.0, dtype=input.dtype), input)


def leaky_relu(
    input: np.ndarray, slope: float | np.ndarray, cache: CacheType | None = None
):
    return np.maximum(np.array(0.0, dtype=input.dtype), input) + slope * np.minimum(
        np.array(0.0, dtype=input.dtype), input
    )


def tanh(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return np.tanh(input)


def sigmoid(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    # For numerical stability implement sigmoid with respect to the
    # sign of input.
    mask = input >= 0
    sig = np.zeros_like(input)
    sig[mask] = 1.0 / (1.0 + np.exp(-input[mask]))
    sig[~mask] = np.exp(input[~mask]) / (1.0 + np.exp(input[~mask]))
    return sig


def softplus(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    # See: https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    return np.log1p(np.exp(-np.abs(input))) + np.maximum(input, 0.0)


def gelu(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return input * (1 + erf(input / np.sqrt(2))) / 2


def softmax(
    input: np.ndarray, *, axis: int = -1, cache: CacheType | None = None
) -> np.ndarray:
    write_into_cache(cache, "axis", axis)
    input_tensor = input - np.max(input, axis=axis, keepdims=True)
    e = np.exp(input_tensor)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s


# Reduction ops
def reduce_mean(
    input: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.mean(input, axis=axis, keepdims=keepdim)


def reduce_sum(
    input: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.sum(input, axis=axis, keepdims=keepdim)


def reduce_max(
    input: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.max(input, axis=axis, keepdims=keepdim)


def reduce_argmax(
    input: np.ndarray,
    *,
    axis: int | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.argmax(input, axis=axis, keepdims=keepdim)


def reduce_min(
    input: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.min(input, axis=axis, keepdims=keepdim)


def reduce_argmin(
    input: np.ndarray,
    *,
    axis: int | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.argmin(input, axis=axis, keepdims=keepdim)


def reduce_prod(
    input: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.prod(input, axis=axis, keepdims=keepdim)


def variance(
    input: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    correction: float = 0.0,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.var(input, axis=axis, ddof=correction, keepdims=keepdim)


# NN ops
def conv1d(
    input: np.ndarray,
    kernel: np.ndarray,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
    cache: CacheType | None = None,
) -> np.ndarray:
    if dilation != 1:
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            f"Currently, the Numpy backend for conv2d only supports a dilation of 1."
        )
    n, c, w = input.shape
    *_, w_k = kernel.shape
    out_w = (w - w_k + sum(padding)) // stride + 1
    submatrices = get_submatrices1d(input, (n, c, out_w), w_k, padding, stride)
    return np.einsum("niwl,oil->now", submatrices, kernel)


def conv1d_bias(
    input: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
    cache: CacheType | None = None,
) -> np.ndarray:
    return (
        conv1d(
            input=input,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            cache=cache,
        )
        + bias
    )


def conv2d(
    input: np.ndarray,
    kernel: np.ndarray,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    cache: CacheType | None = None,
) -> np.ndarray:
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
    _, _, h_k, w_k = kernel.shape
    out_h = (h - h_k + sum(_padding[0])) // stride[0] + 1
    out_w = (w - w_k + sum(_padding[1])) // stride[1] + 1
    submatrices = get_submatrices2d(
        input, (n, c, out_h, out_w), h_k, w_k, _padding, stride[0]
    )
    return np.einsum("nihwkl,oikl->nohw", submatrices, kernel)


def conv2d_bias(
    input: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    cache: CacheType | None = None,
) -> np.ndarray:
    return (
        conv2d(
            input=input,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            cache=cache,
        )
        + bias
    )


def max_pool1d(
    input: np.ndarray,
    kernel_size: int,
    stride: int,
    *,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
    cache: CacheType | None = None,
) -> np.ndarray:
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
    input: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    *,
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    cache: CacheType | None = None,
) -> np.ndarray:
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


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: np.ndarray | None = None,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: bool | None = None,
    cache: CacheType | None = None,
):
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
    input: np.ndarray,
    target: np.ndarray,
    weights: list[float] | bool,
    cutoff: np.ndarray,
    *,
    categorical: bool = True,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    write_into_cache(cache, "weights", _weights)
    log: partial | Callable = (
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
    input: np.ndarray,
    target: np.ndarray,
    weights: list[float] | bool,
    cutoff: np.ndarray,
    *,
    categorical: bool = True,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    log: partial | Callable = (
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
    input: np.ndarray,
    target: np.ndarray,
    weights: list[float] | bool,
    *,
    categorical: bool = True,
    cache: CacheType | None = None,
) -> np.ndarray:
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
    input: np.ndarray,
    target: np.ndarray,
    cutoff: np.ndarray,
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    log: partial | Callable = (
        partial(robust_log, cutoff=cutoff, cache=None) if robust else np.log
    )
    if isinstance(pos_weight, bool) and pos_weight:
        pos_weight = calculate_binary_class_weight(target)
    write_into_cache(cache, "pos_weight", pos_weight)
    return -pos_weight * target * log(input) - (1 - target) * log(1 - input)


def binary_cross_entropy_with_logits(
    input: np.ndarray,
    target: np.ndarray,
    cutoff: np.ndarray,
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
    cache: CacheType | None = None,
) -> np.ndarray:
    log: partial | Callable = (
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
    input: np.ndarray,
    target: np.ndarray,
    quantile: np.ndarray,
    cache: CacheType | None = None,
) -> np.ndarray:
    error = target - input
    return np.maximum(quantile * error, (quantile - 1) * error)


def hinge_loss(
    input: np.ndarray, target: np.ndarray, cache: CacheType | None = None
) -> np.ndarray:
    base_hinge = 1.0 - target * input
    write_into_cache(cache, "base_hinge", base_hinge)
    return np.maximum(0.0, base_hinge)


def quad_hinge_loss(
    input: np.ndarray, target: np.ndarray, cache: CacheType | None = None
) -> np.ndarray:
    return hinge_loss(input, target) ** 2


def kl_divergence(
    input: np.ndarray,
    target: np.ndarray,
    cutoff: np.ndarray,
    cache: CacheType | None = None,
) -> np.ndarray:
    log_input1 = robust_log(input, cutoff)
    log_input2 = robust_log(target, cutoff)
    partial_result = log_input2 - log_input1
    write_into_cache(cache, "partial_result", partial_result)
    return target * partial_result


def absolute_error(
    input: np.ndarray, target: np.ndarray, cache: CacheType | None = None
) -> np.ndarray:
    diff = input - target
    write_into_cache(cache, "diff", diff)
    return np.abs(diff)


def primitive_accuracy(
    input1: np.ndarray, input2: np.ndarray, *, cache: CacheType | None = None
) -> np.ndarray:
    prediction = np.argmax(input1, axis=1).reshape(input1.shape[0], 1)
    return np.mean(prediction == input2)


def auc_core(
    input: np.ndarray, label: np.ndarray, *, cache: CacheType | None = None
) -> np.ndarray:
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


def transposed_diag(input: np.ndarray, *, cache: CacheType | None = None) -> np.ndarray:
    return np.diag(input)[:, np.newaxis]


def broadcast_to(
    input: np.ndarray, shape: tuple[int, ...], cache: CacheType | None = None
) -> np.ndarray:
    return np.broadcast_to(input, shape)


def ones_with_zero_diag(
    *args, precision: int, cache: CacheType | None = None
) -> np.ndarray:
    n, m = args
    output = np.ones((n, m)) - np.eye(n, m) if m is not None else np.ones(n) - np.eye(n)

    return handle_data_precision(output, precision=precision)


def eye(*args, precision: int, cache: CacheType | None = None) -> np.ndarray:
    return handle_data_precision(np.eye(*args), precision=precision)


def squeeze(input: np.ndarray, *, cache: CacheType | None = None) -> np.ndarray:
    return np.squeeze(input)


def to_tensor(
    *input: NestedFloatOrIntOrBoolList, precision: int, cache: CacheType | None = None
) -> np.ndarray:
    return np.array(input[0], dtype=get_type(input[0], precision=precision))


def tensor_to_list(
    input: np.ndarray, cache: CacheType | None = None
) -> NestedFloatOrIntOrBoolList:
    return input.tolist()


def primitive_embedding(
    input: np.ndarray, embedding_matrix: np.ndarray, *, cache: CacheType | None = None
) -> np.ndarray:
    return embedding_matrix[input]


def where(
    cond: np.ndarray,
    input1: np.ndarray,
    input2: np.ndarray,
    *,
    cache: CacheType | None = None,
) -> np.ndarray:
    return np.where(cond, input1, input2)


def concat(
    *inputs: np.ndarray, axis: int | None = 0, cache: CacheType | None = None
) -> np.ndarray:
    return np.concatenate([np.array(v) for v in inputs], axis=axis)


def arange(*args, precision: int, cache: CacheType | None = None) -> np.ndarray:
    return handle_data_precision(np.arange(*args), precision)


def flatten(
    input: np.ndarray,
    *,
    start_dim: int = 0,
    end_dim: int = -1,
    cache: CacheType | None = None,
) -> np.ndarray:
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


def stop_gradient(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    return input


def shape(input: np.ndarray, cache: CacheType | None = None) -> tuple[int, ...]:
    return input.shape


def size(
    input: np.ndarray,
    dim: int | tuple[int, ...] | None,
    cache: CacheType | None = None,
) -> int | tuple[int]:
    if dim is None:
        return input.size
    if isinstance(dim, int):
        return input.shape[dim]
    if isinstance(dim, Sequence):
        return tuple(input.shape[idx] for idx in dim)
    else:
        raise ValueError(f"Unexpected dim: {dim}")


def norm_modifier(input: np.ndarray, cache: CacheType | None = None) -> np.ndarray:
    inner_term = ((input - 1.0) % 8) / 8 - 0.5
    write_into_cache(cache, "inner_term", inner_term)
    return 4.0 * (1.0 - 2.0 * np.abs(inner_term)) + 1.0


def distance_matrix(
    left: np.ndarray,
    right: np.ndarray,
    norm: np.ndarray,
    cache: CacheType | None = None,
) -> np.ndarray:
    diffs = left[:, None, :] - right[None, :, :]
    write_into_cache(cache, "diffs", diffs)
    abs_diffs = np.abs(diffs)
    write_into_cache(cache, "abs_diffs", abs_diffs)
    powered_abs_diffs = abs_diffs**norm
    write_into_cache(cache, "powered_abs_diffs", powered_abs_diffs)
    return np.sum(powered_abs_diffs, axis=2)


def polynomial_features(
    input: np.ndarray, *, degree: int = 2, cache: CacheType | None = None
) -> np.ndarray:
    samples, dims = input.shape
    identity = np.eye(dims + 1, dims + 1, dtype=int)
    data = np.hstack((np.ones((samples, 1), dtype=input.dtype), input))
    write_into_cache(cache, "data", data)
    powers: Iterator = map(sum, combinations_with_replacement(identity, degree))
    # Skip first element of powers. This is the bias term.
    next(powers)
    write_into_cache(
        cache,
        "powers",
        copy.deepcopy(powers),  # type: ignore
    )  # Copy iterator before it is iterated.
    return np.hstack([(data**p).prod(1)[:, np.newaxis] for p in powers])


def tsne_p_joint(
    squared_distances: np.ndarray,
    target_perplexity: np.ndarray,
    threshold: np.ndarray,
    *,
    cache: CacheType | None = None,
) -> np.ndarray:
    """Given a data matrix X, gives joint probabilities matrix.
    Parameters
    ----------
    squared_distances : np.ndarray
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
    input1: np.ndarray, *, cache: CacheType | None = None
) -> np.ndarray | None:
    try:
        return np.linalg.cholesky(input1)
    except np.linalg.LinAlgError as e:
        logging.info(str(e))
    return None


def gpr_alpha(
    label_mu_diff: np.ndarray,
    L: np.ndarray,
    K_term: np.ndarray,
    *,
    cache: CacheType | None = None,
) -> np.ndarray:
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
    K_term: np.ndarray,
    L: np.ndarray,
    threshold: float,
    *,
    cache: CacheType | None = None,
) -> np.ndarray:
    if L is not None:
        return np.diag(L)
    else:
        return np.clip(np.linalg.eigvalsh(K_term), threshold, None)


def gpr_v_outer(
    K: np.ndarray, K_term: np.ndarray, L: np.ndarray, *, cache: CacheType | None = None
) -> np.ndarray:
    if L is not None:
        v = slin.solve_triangular(L, K, lower=True)
        v_outer = v.T @ v
    else:
        v_outer = K.T @ np.linalg.lstsq(K_term, K)[0]
    return v_outer


def isnan(input: np.ndarray, *, cache: CacheType | None = None):
    return np.isnan(input)


def nan_to_num(
    input,
    nan: float,
    posinf: float | None,
    neginf: float | None,
    *,
    cache: CacheType | None = None,
):
    return np.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)  #  type: ignore


def astype(input: np.ndarray, dtype: core.Dtype | int) -> np.ndarray:
    return handle_data_dtype(input, dtype)


def dtype(input: np.ndarray) -> core.Dtype:
    return getattr(core, str(input.dtype))


def logical_xor(
    left: np.ndarray, right: np.ndarray, cache: CacheType | None = None
) -> np.ndarray:
    return np.logical_xor(left, right)


def split(
    input: np.ndarray,
    split_size: int | list[int],
    axis: int = 0,
    cache: CacheType | None = None,
):
    return np.stack(np.split(input, split_size, axis=axis))


def pad(
    input: np.ndarray,
    pad_width: tuple[tuple[int, int], ...],
    cache: CacheType | None = None,
):
    return np.pad(input, pad_width)


array_creation_funcs = [
    "arange",
    "to_tensor",
    "make_array",
    "eye",
    "ones_with_zero_diag",
]
primitive_func_dict = {
    key: fn for key, fn in globals().items() if callable(fn)
} | common_primitive_func_dict
