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

import math
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from itertools import combinations_with_replacement

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .... import core
from ....utils.type_utils import is_int_tuple_tuple, is_tuple_int
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
    primitive_embedding,
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
from . import utils

AxisType = None | int | Sequence[int]

__all__ = [
    "flatten",
    "partial",
    "abs",
    "exp",
    "sqrt",
    "log",
    "sign",
    "robust_sqrt",
    "robust_log",
    "robust_power",
    "stable_reciprocal",
    "sin",
    "cos",
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "softplus",
    "gelu",
    "softmax",
    "reduce_mean",
    "reduce_sum",
    "reduce_min",
    "reduce_max",
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
    "absolute_error",
    "kl_divergence",
    "quantile_loss",
    "hinge_loss",
    "quad_hinge_loss",
    "eye",
    "ones_with_zero_diag",
    "transposed_diag",
    "broadcast_to",
    "squeeze",
    "shape",
    "size",
    "primitive_accuracy",
    "to_tensor",
    "tensor_to_list",
    "arange",
    "concat",
    "matrix_concat",
    "where",
    "stop_gradient",
    "eigvalsh",
    "cholesky",
    "norm_modifier",
    "distance_matrix",
    "tsne_p_joint",
    "polynomial_features",
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
    "add",
    "subtract",
    "multiplication",
    "divide",
    "floor_divide",
    "shift_left",
    "shift_right",
    "power",
    "squared_error",
    "minus",
    "transpose",
    "swapaxes",
    "square",
    "tensor_slice",
    "buffer",
    "permute_tensor",
    "reshape",
    "item",
    "scalar_item",
    "tensor_item",
    "sequence_slice",
    "union",
    "length",
    "cartesian_diff",
    "primitive_embedding",
    "to_tuple",
    "to_list",
    "padding_converter_1d",
    "padding_converter_2d",
    "stride_converter",
    "tuple_converter",
    "common_primitive_func_dict",
    "reduce_argmin",
    "reduce_argmax",
]


# Ops
def abs(input: mx.array) -> mx.array:
    return mx.abs(input)


def exp(input: mx.array) -> mx.array:
    return mx.exp(input)


def sqrt(input: mx.array) -> mx.array:
    return mx.sqrt(input)


def log(input: mx.array) -> mx.array:
    return mx.log(input)


def sign(input: mx.array) -> mx.array:
    return mx.sign(input)


def flatten(input: mx.array, start_dim: int = 0, end_dim: int = -1) -> mx.array:
    return mx.flatten(input, start_axis=start_dim, end_axis=end_dim)


def robust_sqrt(input: mx.array, cutoff: mx.array) -> mx.array:
    input = mx.abs(input)
    cond = input < cutoff
    return mx.where(
        cond,
        input * mx.reciprocal(mx.sqrt(mx.array(cutoff))),
        mx.sqrt(mx.where(cond, 1, input)),
    )


def robust_log(input: mx.array, cutoff: mx.array) -> mx.array:
    input = mx.abs(input)
    cond = input < cutoff
    return mx.where(
        cond,
        mx.log(cutoff) + (input / cutoff) - 1.0,
        mx.log(mx.where(cond, 1, input)),
    )


def robust_power(base: mx.array, exponent: mx.array, threshold: mx.array) -> mx.array:
    # TODO: Handle without using broadcast
    cond = mx.stop_gradient(
        (
            exponent
            < (
                1
                - mx.log(threshold)
                / (
                    mx.log(threshold)
                    + mx.log(mx.abs(mx.stop_gradient(base)) / threshold)
                )
            )
        )
        & (mx.abs(base) < threshold)
    )
    broadcasted_shape = np.broadcast_shapes(base.shape, exponent.shape)
    base = mx.broadcast_to(base, broadcasted_shape)
    exponent = mx.broadcast_to(exponent, broadcasted_shape)

    return mx.where(
        cond,
        (1 / threshold) * mx.abs(base),
        mx.abs(base) ** mx.where(cond, 1, exponent),
    )


def stable_reciprocal(input: mx.array, cutoff: mx.array) -> mx.array:
    # cutoff = cutoff.item()
    cond = mx.abs(input) < cutoff
    return mx.where(
        cond,
        -input / mx.square(cutoff)
        + (2 / cutoff) * (mx.sign(input) + (1 - mx.sign(mx.abs(input)))),
        mx.reciprocal(mx.where(cond, 1, input)),
    )


# Non linearity funcs
def sin(input: mx.array) -> mx.array:
    return mx.sin(input)


def cos(input: mx.array) -> mx.array:
    return mx.cos(input)


def tanh(input: mx.array) -> mx.array:
    return mx.tanh(input)


def relu(input: mx.array) -> mx.array:
    return nn.relu(input)


def leaky_relu(input: mx.array, slope: mx.array) -> mx.array:
    return nn.leaky_relu(input, slope)


def sigmoid(input: mx.array) -> mx.array:
    return mx.sigmoid(input)


def softplus(input: mx.array) -> mx.array:
    return nn.softplus(input)


def gelu(input: mx.array) -> mx.array:
    return nn.gelu(input)


def softmax(input: mx.array, *, axis: int = -1) -> mx.array:
    return mx.softmax(input, axis=axis)


def reduce_mean(
    input: mx.array, *, axis: AxisType = None, keepdim: bool = False
) -> mx.array:
    return mx.mean(input, axis=axis, keepdims=keepdim)


def reduce_sum(
    input: mx.array, *, axis: AxisType = None, keepdim: bool = False
) -> mx.array:
    return mx.sum(input, axis=axis, keepdims=keepdim)


def reduce_min(
    input: mx.array, *, axis: AxisType = None, keepdim: bool = False
) -> mx.array:
    return mx.min(input, axis=axis, keepdims=keepdim)


def reduce_argmin(
    input: mx.array, *, axis: AxisType = None, keepdim: bool = False
) -> mx.array:
    return mx.argmin(input, axis=axis, keepdims=keepdim)


def reduce_max(
    input: mx.array, *, axis: AxisType = None, keepdim: bool = False
) -> mx.array:
    return mx.max(input, axis=axis, keepdims=keepdim)


def reduce_argmax(
    input: mx.array, *, axis: AxisType = None, keepdim: bool = False
) -> mx.array:
    return mx.argmax(input, axis=axis, keepdims=keepdim)


def reduce_prod(
    input: mx.array, *, axis: AxisType = None, keepdim: bool = False
) -> mx.array:
    return mx.prod(input, axis=axis, keepdims=keepdim)


def variance(
    input: mx.array,
    *,
    axis: AxisType = None,
    keepdim: bool = False,
    correction: int | float | None = 0,
) -> mx.array:
    if correction is None:
        correction = 0
    return mx.var(input, axis=axis, ddof=int(correction), keepdims=keepdim)


# NN ops
def conv1d(
    input: mx.array,
    kernel: mx.array,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
) -> mx.array:
    if isinstance(padding, Sequence):
        input = mx.pad(input, [(0, 0), (0, 0), (padding[0], padding[1])])

    # Channel first -> Channel last
    input = mx.swapaxes(input, -2, -1)
    kernel = mx.swapaxes(kernel, -2, -1)
    result = mx.conv1d(
        input,
        kernel,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=1,
    )
    return mx.swapaxes(result, -2, -1)


def conv1d_bias(
    input: mx.array,
    kernel: mx.array,
    bias: mx.array,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
) -> mx.array:
    return (
        conv1d(input, kernel, stride=stride, padding=padding, dilation=dilation) + bias
    )


def conv2d(
    input: mx.array,
    kernel: mx.array,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (
        1,
        1,
    ),
    dilation: tuple[int, int] = (1, 1),
) -> mx.array:
    _padding: tuple[int, int]
    if is_int_tuple_tuple(padding):
        input = mx.pad(
            input,
            [
                (0, 0),
                (0, 0),
                (padding[0][0], padding[0][1]),
                (padding[1][0], padding[1][1]),
            ],
        )
        _padding = (0, 0)
    else:
        _padding = padding  # type: ignore

    # Channel first -> Channel last
    input = mx.swapaxes(input, -3, -1)
    kernel = mx.swapaxes(kernel, -3, -1)
    result = mx.conv2d(
        input,
        kernel,
        stride=stride,
        padding=_padding,
        dilation=dilation,
        groups=1,
    )
    return mx.swapaxes(result, -3, -1)


def conv2d_bias(
    input: mx.array,
    kernel: mx.array,
    bias: mx.array,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> mx.array:
    return (
        conv2d(input, kernel, stride=stride, padding=padding, dilation=dilation) + bias
    )


def max_pool1d(
    input: mx.array,
    kernel_size: int = 2,
    stride: int = 2,
    *,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
) -> mx.array:
    if dilation != 1:
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            f"Currently, the MLX backend for Maxpool1d only supports a dilation of 1."
        )
    n, c, w = input.shape
    out_w = (
        w - kernel_size + sum(padding) if isinstance(padding, tuple) else 2 * padding
    ) // stride + 1
    submatrices = utils.get_submatrices1d(
        input, (n, c, out_w), kernel_size, padding, stride
    )
    return mx.max(submatrices, axis=3)


def max_pool2d(
    input: mx.array,
    *,
    kernel_size: tuple[int, int] = (2, 2),
    stride: int | tuple[int, int] = (2, 2),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> mx.array:
    if dilation != (1, 1):
        raise NotImplementedError(
            f"Dilation of {dilation} is not supported. "
            f"Currently, the MLX backend for Maxpool2d only supports a dilation of 1."
        )

    # Create a new variable for the normalized padding
    normalized_padding: tuple[tuple[int, int], tuple[int, int]]

    if is_tuple_int(padding):
        normalized_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        # TODO: This is now guaranteed to be the correct type
        normalized_padding = padding  # type: ignore

    if isinstance(stride, int):
        stride = (stride, stride)

    n, c, h, w = input.shape
    out_h = (h - kernel_size[0] + sum(normalized_padding[0])) // stride[0] + 1
    out_w = (w - kernel_size[1] + sum(normalized_padding[1])) // stride[1] + 1
    submatrices = utils.get_submatrices2d(
        input,
        (n, c, out_h, out_w),
        kernel_size[0],
        kernel_size[1],
        normalized_padding,
        stride[0],
    )
    return mx.max(submatrices, axis=(4, 5))


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    attn_mask: mx.array | None = None,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | int | None = None,
):
    if dropout_p != 0.0:
        raise RuntimeError(
            "Currently MLX scaled_dot_product_attention only support dropout_p 0"
        )

    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = mx.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = mx.tril(mx.ones((L, S), dtype=mx.bool_), k=0)  # type: ignore
        attn_bias = mx.where(temp_mask, attn_bias, float("-inf"))
        # attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.astype(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == bool:
            attn_bias = mx.where(attn_mask, attn_bias, float("-inf"))
            # attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.swapaxes(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = nn.softmax(attn_weight, axis=-1)
    return attn_weight @ value


# Loss funcs
def cross_entropy(
    input: mx.array,
    target: mx.array,
    weights: list[float] | bool,
    cutoff: mx.array,
    *,
    categorical: bool = True,
    robust: bool = False,
) -> mx.array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else mx.log
    _weights = utils.calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    if categorical:
        if not mx.issubdtype(target.dtype, mx.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )
        return (
            -log(mx.take_along_axis(input, target[:, None], axis=1)[:, 0])
            * _weights[target]
        )
    return -mx.sum(target * log(input) * _weights, axis=1)


def cross_entropy_with_logits(
    input: mx.array,
    target: mx.array,
    weights: list[float] | bool,
    cutoff: mx.array,
    *,
    categorical: bool = True,
    robust: bool = False,
) -> mx.array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else mx.log
    _weights = utils.calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    if categorical:
        if not mx.issubdtype(target.dtype, mx.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )

        logits_max = mx.max(input, axis=1, keepdims=True)
        input -= mx.stop_gradient(logits_max)
        label_logits = mx.take_along_axis(input, target[:, None], axis=1)[:, 0]
        log_normalizers = log(mx.sum(mx.exp(input), axis=1))
        return (log_normalizers - label_logits) * _weights[target]
    return -mx.sum(
        target * utils.log_softmax(input, log, robust, axis=1) * _weights, axis=1
    )


def cross_entropy_with_log_probs(
    input: mx.array,
    target: mx.array,
    weights: list[float] | bool,
    *,
    categorical: bool = True,
) -> mx.array:
    _weights = utils.calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    if categorical:
        if not mx.issubdtype(target.dtype, mx.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )

        return (
            -mx.take_along_axis(input, target[:, None], axis=1)[:, 0] * _weights[target]
        )
    return -mx.sum(target * input * _weights, axis=1)


def binary_cross_entropy(
    input: mx.array,
    target: mx.array,
    cutoff: mx.array,
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
) -> mx.array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else mx.log
    if isinstance(pos_weight, bool) and pos_weight:
        _pos_weight = utils.calculate_binary_class_weight(target)
    else:
        _pos_weight = pos_weight

    return -_pos_weight * target * log(input) - (1 - target) * log(1 - input)


def binary_cross_entropy_with_logits(
    input: mx.array,
    target: mx.array,
    cutoff: mx.array,
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
) -> mx.array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else mx.log
    _pos_weight: mx.array | float

    if isinstance(pos_weight, bool):
        _pos_weight = (
            utils.calculate_binary_class_weight(mx.sigmoid(target))
            if pos_weight
            else 1.0
        )
    else:
        _pos_weight = pos_weight

    if _pos_weight != 1.0:
        log_weight = (_pos_weight - 1) * (target) + 1
        loss = (1 - target) * input - (
            log_weight * utils.log_sigmoid(input, log, robust)
        )
    else:
        loss = (1 - target) * input - utils.log_sigmoid(input, log, robust)

    return loss


def absolute_error(input: mx.array, target: mx.array) -> mx.array:
    return mx.abs(input - target)


def kl_divergence(input: mx.array, target: mx.array, cutoff: mx.array) -> mx.array:
    return target * (robust_log(target, cutoff) - robust_log(input, cutoff))


def quantile_loss(input: mx.array, target: mx.array, quantile: mx.array) -> mx.array:
    error = target - input
    return mx.maximum(quantile * error, (quantile - 1) * error)


def hinge_loss(input: mx.array, target: mx.array) -> mx.array:
    return mx.maximum(0.0, 1.0 - target * input)


def quad_hinge_loss(input: mx.array, target: mx.array) -> mx.array:
    return hinge_loss(input, target) ** 2


def eye(N: int, M: int | None, *, device: str, precision: int) -> mx.array:
    return utils.handle_data_precision(mx.eye(N, M), precision)


def ones_with_zero_diag(N: int, M: int | None, device: str, precision: int) -> mx.array:
    output = mx.ones(N) - mx.eye(N) if M is None else mx.ones((N, M)) - mx.eye(N, M)
    return utils.handle_data_precision(output, precision)


def transposed_diag(input: mx.array) -> mx.array:
    return mx.diag(input)[:, None]


def broadcast_to(input: mx.array, shape: tuple[int, ...]) -> mx.array:
    return mx.broadcast_to(input, shape)


def squeeze(input: mx.array) -> mx.array:
    return mx.squeeze(input)


def shape(input: mx.array) -> tuple[int, ...]:
    return input.shape


def size(input: mx.array, dim: int | tuple[int, ...] | None) -> int | tuple[int]:
    if dim is None:
        return input.size
    if isinstance(dim, int):
        return input.shape[dim]
    if isinstance(dim, Sequence):
        return tuple(input.shape[idx] for idx in dim)
    else:
        raise ValueError(f"Unexpected dim: {dim}")


def primitive_accuracy(input1: mx.array, input2: mx.array) -> mx.array:
    prediction = mx.argmax(input1, axis=1).reshape(input1.shape[0], 1)
    # MLX return type is array | bool!
    return mx.mean(mx.equal(prediction, input2))


def to_tensor(
    input: NestedFloatOrIntOrBoolList, device: str, precision: int
) -> mx.array:
    return mx.array(input, dtype=utils.get_type(input, precision))


def tensor_to_list(input: mx.array) -> NestedFloatOrIntOrBoolList:
    # MLX return type is object!
    return input.tolist()  # type: ignore


def arange(*args, device: str, precision: int) -> mx.array:
    out = mx.arange(*args)
    return utils.handle_data_precision(out, precision)


def concat(*inputs: mx.array, axis: AxisType = 0) -> mx.array:
    return mx.concatenate([mx.atleast_1d(v) for v in inputs], axis=axis)


def matrix_concat(input1: mx.array, input2: mx.array) -> mx.array:
    return mx.concatenate([input1, input2], axis=input1.ndim - 1)


def where(cond: mx.array, input1: mx.array, input2: mx.array) -> mx.array:
    return mx.where(cond, input1, input2)


def stop_gradient(input: mx.array) -> mx.array:
    return mx.stop_gradient(input)


def eigvalsh(K_term: mx.array, L: mx.array, threshold: float) -> mx.array:
    raise NotImplementedError("eigvalsh is not implemented for MLX backend.")


def cholesky(input: mx.array) -> mx.array:
    raise NotImplementedError("cholesky is not implemented for MLX backend.")


def norm_modifier(input: mx.array) -> mx.array:
    input = input - 1.0
    if input // 8 <= 0:
        input += (1 - input // 8) * 8
    inner_term = (input % 8) / 8 - 0.5
    return 4.0 * (1.0 - 2.0 * mx.abs(inner_term)) + 1.0


def distance_matrix(left: mx.array, right: mx.array, norm: mx.array) -> mx.array:
    diffs = left[:, None, :] - right[None, :, :]
    abs_diffs = mx.abs(diffs)
    powered_abs_diffs = abs_diffs**norm
    powered_dists = mx.sum(powered_abs_diffs, axis=2)
    return powered_dists


def tsne_p_joint(
    squared_distances: mx.array,
    target_perplexity: mx.array,
    threshold: mx.array,
) -> mx.array:
    """Given a data matrix X, gives joint probabilities matrix.
    Parameters
    ----------
    squared_distances : mx.array
        Square of distance matrix of Input data.
    target_perplexity : float
        Desired perplexity value.
    Returns
    -------
    mx.array
        Matrix with entries p_ij: joint probabilities.
    """
    # Get the negative euclidian distances matrix for our data
    negative_dist_sq = (-1) * squared_distances
    # Find optimal sigma for each row of this distances matrix
    # TODO: Find correct types
    sigmas = utils.find_optimal_sigmas(negative_dist_sq, target_perplexity, threshold)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = utils.calc_prob_matrix(negative_dist_sq, sigmas)
    # Go from conditional to joint probabilities matrix in a symmetrical manner.
    P = (p_conditional + p_conditional.T) / (2.0 * p_conditional.shape[0])
    return P


def polynomial_features(input: mx.array, *, degree: int = 2) -> mx.array:
    samples, dims = input.shape
    identity = mx.eye(dims + 1, dims + 1, dtype=input.dtype)
    data = mx.concatenate([mx.ones((samples, 1), dtype=input.dtype), input], axis=1)
    powers: Iterator = map(sum, combinations_with_replacement(identity, degree))
    # Skip first element of powers. This is the bias term.
    next(powers)
    return mx.concatenate(
        [
            mx.expand_dims(utils.polynomial_features_helper(data, p).prod(1), 1)
            for p in powers
        ],
        axis=1,
    )


def isnan(input):
    return mx.isnan(input)


def nan_to_num(
    input,
    nan: int | float | None,
    posinf: int | float | None,
    neginf: int | float | None,
):
    raise NotImplementedError("nan_to_num is not implemented in mlx!")
    # return mx.nan_to_num(input, nan = nan, posinf = posinf, neginf = neginf)


def astype(input: mx.array, dtype: core.Dtype | int) -> mx.array:
    return utils.handle_data_dtype(input, dtype)


def dtype(input: mx.array) -> core.Dtype:
    return getattr(core, str(input.dtype).split(".")[-1])


def logical_xor(left: mx.array, right: mx.array):
    return mx.bitwise_xor(left, right)


def split(input: mx.array, split_size: int | list[int], axis: int = 0):
    return mx.stack(mx.split(input, split_size, axis=axis))  # type: ignore


def pad(input: mx.array, pad_width: tuple[tuple[int, int], ...]):
    return mx.pad(input, pad_width)


array_creation_funcs = ["arange", "to_tensor", "eye", "ones_with_zero_diag"]
primitive_func_dict = common_primitive_func_dict = {
    key: fn for key, fn in globals().items() if callable(fn)
} | common_primitive_func_dict
