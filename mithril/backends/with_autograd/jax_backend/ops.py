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

from collections.abc import Callable, Iterator, Sequence
from functools import partial
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp
import jax.scipy.linalg as slin
from jax import lax, vmap
from jax import nn as functionals

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
from .utils import (
    broadcast_to_highest,
    calc_prob_matrix,
    calculate_binary_class_weight,
    calculate_cross_entropy_class_weights,
    calculate_tpr_fpr,
    find_optimal_sigmas,
    get_device,
    get_type,
    handle_data_dtype,
    handle_data_precision,
    log_sigmoid,
    log_softmax,
    many_to_one_inference_helper,
    polynomial_features_helper,
    robust_log_helper,
    robust_power_helper,
    robust_sqrt_helper,
    stable_reciprocal_helper,
    vmapper,
)

AxisType = None | int | Sequence[int]

__all__ = [
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
    "many_to_one_inference",
    "cross_entropy",
    "cross_entropy_with_logits",
    "cross_entropy_with_log_probs",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "absolute_error",
    "quantile_loss",
    "hinge_loss",
    "quad_hinge_loss",
    "kl_divergence",
    "eye",
    "ones_with_zero_diag",
    "transposed_diag",
    "primitive_accuracy",
    "auc_core",
    "squeeze",
    "shape",
    "size",
    "flatten",
    "concat",
    "broadcast_to",
    "matrix_concat",
    "to_tensor",
    "tensor_to_list",
    "arange",
    "where",
    "stop_gradient",
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
    "array_creation_funcs",
    "primitive_func_dict",
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
    "unique",
    "trapezoid",
]


# Ops
def abs(input: jax.Array) -> jax.Array:
    return jnp.abs(input)


def exp(input: jax.Array) -> jax.Array:
    return jnp.exp(input)


def sqrt(input: jax.Array) -> jax.Array:
    return jnp.sqrt(input)


def log(input: jax.Array) -> jax.Array:
    return jnp.log(input)


def sign(input: jax.Array) -> jax.Array:
    return jnp.sign(input)


def robust_sqrt(input: jax.Array, cutoff: jax.Array) -> jax.Array:
    # v_mapped_func= jax.vmap(partial(robust_log_helper, threshold = cutoff))
    v_mapped_func = vmapper(
        partial(robust_sqrt_helper, threshold=cutoff), len(input.shape) - 1
    )
    # It is required to have 2D arrays for doubled vmap operations.
    # So first make input array as 2D and finally convert it to its
    # original shape.
    return v_mapped_func(jnp.atleast_1d(input)).reshape(input.shape)


# NOTE: We wrote the stabilized log in order to handle
# undefined points (log(0) = -inf in this case),
# further testing should be done about performance.
def robust_log(input: jax.Array, cutoff: jax.Array) -> jax.Array:
    # v_mapped_func= jax.vmap(partial(robust_log_helper, threshold = cutoff))
    v_mapped_func = vmapper(
        partial(robust_log_helper, threshold=cutoff), len(input.shape) - 1
    )
    # It is required to have 2D arrays for doubled vmap operations.
    # So first make input array as 2D and finally convert it to its
    # original shape.
    return v_mapped_func(jnp.atleast_1d(input)).reshape(input.shape)


# NOTE: We wrote robust_power in order to avoid getting
# its derivative infinitive at 0 point ( d/dx(x**y) = inf at x = 0)
# further testing should be done about performance.
def robust_power(
    base: jax.Array, exponent: jax.Array, threshold: jax.Array
) -> jax.Array:
    in_1, in_2, final_shape = broadcast_to_highest(base, exponent)
    v_mapped_func = vmapper(
        partial(robust_power_helper, threshold=threshold), len(final_shape) - 1
    )

    # Guarantee inputs has at least 2 dimensions.
    return jnp.reshape(
        v_mapped_func(jnp.atleast_1d(in_1), jnp.atleast_1d(in_2)), final_shape
    )


# NOTE: We wrote stable reciprocal in order to handle
# undefined points (f(0) = inf in this case),
# futher testing should be done.
def stable_reciprocal(input: jax.Array, cutoff: jax.Array) -> jax.Array:
    v_mapped_func = vmapper(
        partial(stable_reciprocal_helper, threshold=cutoff),
        len(input.shape) - 1,
    )
    return v_mapped_func(jnp.atleast_1d(input)).reshape(input.shape)


# Non linearity funcs
def sin(input: jax.Array) -> jax.Array:
    return jnp.sin(input)


def cos(input: jax.Array) -> jax.Array:
    return jnp.cos(input)


def tanh(input: jax.Array) -> jax.Array:
    return jnp.tanh(input)


def relu(input: jax.Array) -> jax.Array:
    return functionals.relu(input)


def leaky_relu(input: jax.Array, slope: jax.Array):
    return functionals.leaky_relu(input, slope)


def sigmoid(input: jax.Array) -> jax.Array:
    return functionals.sigmoid(input)


def softplus(input: jax.Array) -> jax.Array:
    return functionals.softplus(input)


def gelu(input: jax.Array) -> jax.Array:
    return functionals.gelu(input, approximate=False)


def softmax(input: jax.Array, *, axis: int = -1) -> jax.Array:
    return functionals.softmax(input, axis=axis)


# Reduction ops
def reduce_mean(
    input: jax.Array, *, axis: AxisType = None, keepdim: bool = False
) -> jax.Array:
    return jnp.mean(input, axis=axis, keepdims=keepdim)


def reduce_sum(
    input: jax.Array, *, axis: AxisType = None, keepdim: bool = False
) -> jax.Array:
    return jnp.sum(input, axis=axis, keepdims=keepdim)


def reduce_min(
    input: jax.Array, *, axis: AxisType = None, keepdim: bool = False
) -> jax.Array:
    return jnp.min(input, axis=axis, keepdims=keepdim)


def reduce_argmin(
    input: jax.Array, *, axis: int | None = None, keepdim: bool = False
) -> jax.Array:
    return jnp.argmin(input, axis=axis, keepdims=keepdim)


def reduce_max(
    input: jax.Array, *, axis: AxisType = None, keepdim: bool = False
) -> jax.Array:
    return jnp.max(input, axis=axis, keepdims=keepdim)


def reduce_argmax(
    input: jax.Array, *, axis: int | None = None, keepdim: bool = False
) -> jax.Array:
    return jnp.argmax(input, axis=axis, keepdims=keepdim)


def reduce_prod(
    input: jax.Array, *, axis: AxisType = None, keepdim: bool = False
) -> jax.Array:
    return jnp.prod(input, axis=axis, keepdims=keepdim)


def variance(
    input: jax.Array,
    *,
    axis: AxisType = None,
    keepdim: bool = False,
    correction: int = 0,
) -> jax.Array:
    return jnp.var(input, axis=axis, ddof=correction, keepdims=keepdim)


# NN ops
def conv1d(
    input: jax.Array,
    kernel: jax.Array,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
) -> jax.Array:
    return lax.conv_general_dilated(
        input,
        kernel,
        (stride,),
        (padding,),
        feature_group_count=1,
        lhs_dilation=(dilation,),
    )


def conv1d_bias(
    input: jax.Array,
    kernel: jax.Array,
    bias: jax.Array,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
) -> jax.Array:
    return (
        conv1d(input, kernel, stride=stride, padding=padding, dilation=dilation) + bias
    )


def conv2d(
    input: jax.Array,
    kernel: jax.Array,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
):
    _padding_normalized: tuple[tuple[int, int], tuple[int, int]]
    if is_tuple_int(padding):
        _padding_normalized = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        _padding_normalized = padding  # type: ignore

    return lax.conv_general_dilated(
        input,
        kernel,
        stride,
        _padding_normalized,
        lhs_dilation=dilation,
        rhs_dilation=None,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=1,
    )


def conv2d_bias(
    input: jax.Array,
    kernel: jax.Array,
    bias: jax.Array,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
):
    return (
        conv2d(
            input=input,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        + bias
    )


def max_pool1d(
    input: jax.Array,
    kernel_size: int,
    stride: int,
    *,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
) -> jax.Array:
    """Implements torch.nn.functional.max_pool1d in JAX"""

    num_batch_dims = input.ndim - 1
    strides: tuple[int, ...] = (stride,)  # or kernel_size

    strides = (1,) * num_batch_dims + strides
    dims = (1,) * num_batch_dims + (kernel_size,)
    _dilation = (1,) * num_batch_dims + (dilation,)

    is_single_input = False
    if num_batch_dims == 0:
        input = input[None]
        strides = (1,) + strides
        dims = (1,) + dims
        is_single_input = True

    assert input.ndim == len(dims), f"len({input.shape}) != len({dims})"

    _padding = ((0, 0),) * num_batch_dims + (padding,)

    y = lax.reduce_window(input, -jnp.inf, lax.max, dims, strides, _padding, _dilation)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def max_pool2d(
    input: jax.Array,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    *,
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> jax.Array:
    """Implements torch.nn.functional.max_pool2d in JAX"""

    _padding: tuple[tuple[int, int], tuple[int, int]]
    if is_tuple_int(padding):
        _padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        _padding = padding  # type: ignore

    if isinstance(stride, int):
        stride = (stride, stride)

    num_batch_dims = input.ndim - len(kernel_size)
    assert len(kernel_size) == len(
        stride
    ), f"len({kernel_size}) must equal len({stride})"
    _stride = (1,) * num_batch_dims + stride
    dims = (1,) * num_batch_dims + kernel_size
    _dilation = (1,) * num_batch_dims + dilation

    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        input = input[None]
        _stride = (1,) + _stride
        dims = (1,) + dims
        is_single_input = True

    assert input.ndim == len(dims), f"len({input.shape}) != len({dims})"
    assert len(padding) == len(kernel_size), (
        f"padding {padding} must specify pads for same number of dims as "
        f"kernel_size {kernel_size}"
    )
    assert all(
        [len(_padding) == 2 for x in padding]
    ), f"each entry in padding {padding} must be length 2"
    __padding = ((0, 0),) * num_batch_dims + _padding

    y = lax.reduce_window(input, -jnp.inf, lax.max, dims, _stride, __padding, _dilation)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def scaled_dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attn_mask: jax.Array | None = None,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | int | None = None,
):
    if dropout_p != 0.0:
        raise RuntimeError(
            "Currently Jax scaled_dot_product_attention only support dropout_p 0"
        )

    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / jnp.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = jnp.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = jnp.tril(jnp.ones((L, S), dtype=bool))
        attn_bias = jnp.where(temp_mask, attn_bias, float("-inf"))
        # attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.astype(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == bool:
            attn_bias = jnp.where(attn_mask, attn_bias, float("-inf"))
            # attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.swapaxes(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax(attn_weight, axis=-1)
    return attn_weight @ value


def many_to_one_inference(
    input: jax.Array,
    prev_hidden: jax.Array,
    length: jax.Array,  # TODO: Discuss this type!
    w_ih: jax.Array,
    w_hh: jax.Array,
    w_ho: jax.Array,
    bias_h: jax.Array,
    bias_o: jax.Array,
) -> jax.Array:
    partial_fun = partial(
        many_to_one_inference_helper,
        w_ih=w_ih,
        w_hh=w_hh,
        w_ho=w_ho,
        bias_h=bias_h,
        bias_o=bias_o,
    )
    _, out = jax.lax.scan(partial_fun, prev_hidden, input)
    return out[length - 1]


# Loss funcs
def cross_entropy(
    input: jax.Array,
    target: jax.Array,
    weights: list[float] | bool,
    cutoff: jax.Array,
    *,
    categorical: bool = True,
    robust: bool = False,
) -> jax.Array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else jnp.log
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )

    if categorical:
        if not jnp.issubdtype(target.dtype, jnp.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )

        return (
            -log(jnp.take_along_axis(input, target[:, None], axis=1)[:, 0])
            * _weights[target]
        )
    return -jnp.sum(target * log(input) * _weights, axis=1)


def cross_entropy_with_logits(
    input: jax.Array,
    target: jax.Array,
    weights: list[float] | bool,
    cutoff: jax.Array,
    *,
    categorical: bool = True,
    robust: bool = False,
) -> jax.Array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else jnp.log
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    if categorical:
        if not jnp.issubdtype(target.dtype, jnp.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )
        logits_max = jnp.max(input, axis=1, keepdims=True)
        input -= jax.lax.stop_gradient(logits_max)
        label_logits = jnp.take_along_axis(input, target[:, None], axis=1)[:, 0]
        log_normalizers = log(jnp.sum(jnp.exp(input), axis=1))
        return (log_normalizers - label_logits) * _weights[target]
    return -jnp.sum(target * log_softmax(input, log, robust, axis=1) * _weights, axis=1)


def cross_entropy_with_log_probs(
    input: jax.Array,
    target: jax.Array,
    weights: list[float] | bool,
    *,
    categorical: bool = True,
) -> jax.Array:
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    if categorical:
        if not jnp.issubdtype(target.dtype, jnp.integer):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )

        return (
            -jnp.take_along_axis(input, target[:, None], axis=1)[:, 0]
            * _weights[target]
        )
    return -jnp.sum(target * input * _weights, axis=1)


def binary_cross_entropy(
    input: jax.Array,
    target: jax.Array,
    cutoff: jax.Array,
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
) -> jax.Array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else jnp.log

    _pos_weight: jax.Array | float
    if isinstance(pos_weight, bool):
        _pos_weight = calculate_binary_class_weight(target) if pos_weight else 1.0
    else:
        _pos_weight = pos_weight

    return -_pos_weight * target * log(input) - (1 - target) * log(1 - input)


def binary_cross_entropy_with_logits(
    input: jax.Array,
    target: jax.Array,
    cutoff: jax.Array,
    *,
    pos_weight: float | bool = 1.0,
    robust: bool = False,
) -> jax.Array:
    log: partial | Callable = partial(robust_log, cutoff=cutoff) if robust else jnp.log

    _pos_weight: jax.Array | float
    if isinstance(pos_weight, bool):
        _pos_weight = (
            calculate_binary_class_weight(functionals.sigmoid(target))
            if pos_weight
            else 1.0
        )
    else:
        _pos_weight = pos_weight

    log_weight = (_pos_weight - 1) * (target) + 1
    loss = (1 - target) * input - (log_weight * log_sigmoid(input, log, robust))

    return loss


def absolute_error(input: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.abs(input - target)


def quantile_loss(
    input: jax.Array, target: jax.Array, quantile: jax.Array
) -> jax.Array:
    error = target - input
    return jnp.maximum(quantile * error, (quantile - 1) * error)


def hinge_loss(input: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.maximum(0.0, 1.0 - target * input)


def quad_hinge_loss(input: jax.Array, target: jax.Array) -> jax.Array:
    return hinge_loss(input, target) ** 2


def kl_divergence(input: jax.Array, target: jax.Array, cutoff: jax.Array) -> jax.Array:
    return target * (robust_log(target, cutoff) - robust_log(input, cutoff))


def eye(N: int, M: int | None, *, device: str, precision: int) -> jax.Array:
    with jax.default_device(get_device(device)):
        return handle_data_precision(jnp.eye(N, M), precision)


def ones_with_zero_diag(
    N: int, M: int | None, device: str, precision: int
) -> jax.Array:
    output = jnp.ones(N) - jnp.eye(N) if M is None else jnp.ones((N, M)) - jnp.eye(N, M)

    with jax.default_device(get_device(device)):
        return handle_data_precision(output, precision)


def transposed_diag(input: jax.Array) -> jax.Array:
    return jnp.diag(input)[:, None]


# TODO: Remove this?
def primitive_accuracy(input1: jax.Array, input2: jax.Array) -> jax.Array:
    prediction = jnp.argmax(input1, axis=1).reshape(input1.shape[0], 1)
    return jnp.mean(prediction == input2)


def auc_core(input: jax.Array, label: jax.Array) -> jax.Array:
    if input.ndim > 1:
        raise ValueError(f"Input should be 1D array, but given '{input.ndim}D'")
    if label.ndim > 1:
        raise ValueError(f"Label should be 1D array, but given '{label.ndim}D'")
    # TODO: Instead of looping all inputs, loop through unique values!
    # But currently jnp.unique cannot be jitted find workaround.
    tprs, fprs = jax.vmap(lambda threshold: calculate_tpr_fpr(threshold, input, label))(
        jnp.flip(jnp.sort(input))
    )

    return jnp.stack([tprs, fprs])


def unique(input: jax.Array) -> jax.Array:
    return jnp.unique(input)


def trapezoid(y: jax.Array, x: jax.Array) -> jax.Array:
    return jax.scipy.integrate.trapezoid(y, x)


def squeeze(input: jax.Array) -> jax.Array:
    return jnp.squeeze(input)


def shape(input: jax.Array) -> tuple[int, ...]:
    return input.shape


def size(input: jax.Array, dim: int | tuple[int, ...] | None) -> int | tuple[int, ...]:
    if dim is None:
        return input.size
    if isinstance(dim, int):
        return input.shape[dim]
    if isinstance(dim, tuple):
        return tuple(input.shape[idx] for idx in dim)
    else:
        raise ValueError(f"Unexpected dim: {dim}")


def flatten(input: jax.Array, *, start_dim: int = 0, end_dim: int = -1) -> jax.Array:
    """Flattens a JAX array akin to torch.flatten"""
    start_dim = (
        int(start_dim == -1) * len(input.shape) + int(start_dim != -1) * start_dim
    )
    end_dim = int(end_dim == -1) * len(input.shape) + int(end_dim != -1) * end_dim
    shape: tuple[int, ...] = (
        input.shape[:start_dim]
        + int(start_dim != end_dim) * (-1,)
        + input.shape[end_dim + 1 :]
    )

    if len(shape) == 0:
        shape = (1,)

    return jnp.reshape(input, shape)


def concat(*inputs: jax.Array, axis: int | None = 0) -> jax.Array:
    return jnp.concatenate([jnp.atleast_1d(v) for v in inputs], axis=axis)


def broadcast_to(input: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return jnp.broadcast_to(input, shape)


def matrix_concat(input1: jax.Array, input2: jax.Array) -> jax.Array:
    return jnp.concatenate((input1, input2), axis=input1.ndim - 1)


def to_tensor(
    input: NestedFloatOrIntOrBoolList, device: str, precision: int
) -> jax.Array:
    with jax.default_device(get_device(device)):
        return jnp.array(input, dtype=get_type(input, precision))


def tensor_to_list(input: jax.Array) -> NestedFloatOrIntOrBoolList:
    return input.tolist()


def arange(*args, device: str, precision: int) -> jax.Array:
    with jax.default_device(get_device(device)):
        return handle_data_precision(jnp.arange(*args), precision)


def where(cond: jax.Array, input1: jax.Array, input2: jax.Array) -> jax.Array:
    return jnp.where(cond, input1, input2)


def stop_gradient(input: jax.Array) -> jax.Array:
    return jax.lax.stop_gradient(input)


# Other stuff
def norm_modifier(input: jax.Array) -> jax.Array:
    inner_term = ((input - 1.0) % 8) / 8 - 0.5
    return 4.0 * (1.0 - 2.0 * jnp.abs(inner_term)) + 1.0


def distance_matrix(left: jax.Array, right: jax.Array, norm: jax.Array) -> jax.Array:
    diffs = left[:, None, :] - right[None, :, :]
    abs_diffs = jnp.abs(diffs)
    powered_abs_diffs = abs_diffs**norm
    powered_dists = jnp.sum(powered_abs_diffs, axis=2)
    return powered_dists


def polynomial_features(input: jax.Array, *, degree: int = 2) -> jax.Array:
    samples, dims = input.shape
    identity = jnp.eye(dims + 1, dims + 1, dtype=input.dtype)
    data = jnp.hstack((jnp.ones((samples, 1), dtype=input.dtype), input))
    powers: Iterator = map(sum, combinations_with_replacement(identity, degree))
    # Skip first element of powers. This is the bias term.
    next(powers)
    return jnp.hstack(
        [
            vmap(vmap(polynomial_features_helper), in_axes=(0, None))(data, p).prod(1)[
                :, jnp.newaxis
            ]
            for p in powers
        ]
    )


def tsne_p_joint(
    squared_distances: jax.Array,
    target_perplexity: jax.Array,
    threshold: jax.Array,
) -> jax.Array:
    """Given a data matrix X, gives joint probabilities matrix.
    Parameters
    ----------
    squared_distances : jax.Array
        Square of distance matrix of Input data.
    target_perplexity : float
        Desired perplexity value.
    Returns
    -------
    jax.Array
        Matrix with entries p_ij: joint probabilities.
    """
    # Get the negative euclidian distances matrix for our data
    negative_dist_sq = (-1) * squared_distances
    # Find optimal sigma for each row of this distances matrix
    # TODO: Fix wrong types!
    sigmas = find_optimal_sigmas(negative_dist_sq, target_perplexity, threshold)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(negative_dist_sq, sigmas)
    # Go from conditional to joint probabilities matrix in a symmetrical manner.
    P = (p_conditional + p_conditional.T) / (2.0 * p_conditional.shape[0])
    return P


def cholesky(input: jax.Array) -> jax.Array:
    # try:
    return jnp.linalg.cholesky(input)
    # except:
    #     logging.info("CHOLESKY FACTORIZATION DOES NOT EXIST!!!")
    # return None


def gpr_alpha(label_mu_diff: jax.Array, L: jax.Array, K_term: jax.Array) -> jax.Array:
    if L is not None:
        alpha = slin.solve_triangular(
            L.T,
            slin.solve_triangular(L, label_mu_diff, lower=True),
            lower=False,
        )
    else:
        alpha = jnp.linalg.solve(K_term, label_mu_diff)
    return alpha


def eigvalsh(K_term: jax.Array, L: jax.Array, threshold: float) -> jax.Array:
    if L is not None:
        return jnp.diag(L)
    else:
        return jnp.clip(jnp.linalg.eigvalsh(K_term), threshold, None) / 2


def gpr_v_outer(K: jax.Array, K_term: jax.Array, L: jax.Array) -> jax.Array:
    if L is not None:
        v = slin.solve_triangular(L, K, lower=True)
        v_outer = v.T @ v
    else:
        v_outer = K.T @ jnp.linalg.lstsq(K_term, K)[0]
    return v_outer


def isnan(input):
    return jnp.isnan(input)


def nan_to_num(
    input,
    nan: int | float | None,
    posinf: int | float | None,
    neginf: int | float | None,
):
    return jnp.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)  # type: ignore


def astype(input: jax.Array, dtype: core.Dtype | int) -> jax.Array:
    return handle_data_dtype(input, dtype)


def dtype(input: jax.Array) -> core.Dtype:
    return getattr(core.Dtype, str(input.dtype))


def logical_xor(left: jax.Array, right: jax.Array):
    return left ^ right


def split(input: jax.Array, split_size: int | list[int], axis: int = 0):
    return jnp.stack(jnp.split(input, split_size, axis=axis))


def pad(input: jax.Array, pad_width: tuple[tuple[int, int], ...]):
    return jax.numpy.pad(input, pad_width)


array_creation_funcs = ["arange", "to_tensor", "eye", "ones_with_zero_diag"]
primitive_func_dict = common_primitive_func_dict = {
    key: fn for key, fn in globals().items() if callable(fn)
} | common_primitive_func_dict
