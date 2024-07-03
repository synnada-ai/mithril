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

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from itertools import combinations_with_replacement

import torch
import torch.nn.functional as F  # noqa: N812
from torch.distributed._tensor import DeviceMesh, Replicate, distribute_tensor

from .... import core
from ....utils.type_utils import is_int_tuple_tuple
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
    tuple_converter,
    union,
)
from .utils import (
    calc_prob_matrix,
    calculate_binary_class_weight,
    calculate_cross_entropy_class_weights,
    calculate_tpr_fpr,
    find_optimal_sigmas,
    get_type,
    handle_data_dtype,
    handle_data_precision,
    log_sigmoid,
    log_softmax,
)

AxisType = None | int | Sequence[int]

__all__ = [
    "partial",
    "abs",
    "sqrt",
    "log",
    "sign",
    "exp",
    "robust_power",
    "robust_sqrt",
    "robust_log",
    "stable_reciprocal",
    "sin",
    "cos",
    "relu",
    "gelu",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "softplus",
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
    "positional_encoding",
    "scaled_dot_product_attention",
    "cross_entropy",
    "cross_entropy_with_logits",
    "cross_entropy_with_log_probs",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "hinge_loss",
    "quad_hinge_loss",
    "absolute_error",
    "quantile_loss",
    "kl_divergence",
    "eye",
    "transposed_diag",
    "ones_with_zero_diag",
    "primitive_accuracy",
    "auc_core",
    "squeeze",
    "broadcast_to",
    "transpose",
    "where",
    "to_tensor",
    "tensor_to_list",
    "to_parallel",
    "arange",
    "concat",
    "matrix_concat",
    "stop_gradient",
    "flatten",
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
    "lstm_cell",
    "reduce_argmin",
    "reduce_argmax",
    "astype",
    "unique",
    "trapezoid",
]


# Ops
def abs(input: torch.Tensor) -> torch.Tensor:
    return torch.abs(input)


def sqrt(input: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(input)


def log(input: torch.Tensor) -> torch.Tensor:
    return torch.log(input)


def sign(input) -> torch.Tensor:
    return torch.sign(input)


def exp(input: torch.Tensor) -> torch.Tensor:
    return torch.exp(input)


def robust_power(
    base: torch.Tensor, exponent: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    threshold1 = threshold
    result_shape = torch.broadcast_shapes(base.shape, exponent.shape)
    broadcasted_base = torch.broadcast_to(base, result_shape)
    broadcasted_exponent = torch.broadcast_to(exponent, result_shape)
    broadcasted_base = torch.abs(broadcasted_base)
    threshold = torch.broadcast_to(threshold.to(base.dtype), result_shape)
    cond = (
        broadcasted_exponent < (1 - torch.log(threshold) / torch.log(broadcasted_base))
    ) & (broadcasted_base < threshold)
    output = torch.where(
        cond,
        (1 / threshold1) * broadcasted_base,
        broadcasted_base ** torch.where(cond, 1, exponent),
    )
    return output


def robust_sqrt(input: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    input = torch.abs(input)
    inds = input < cutoff
    output = torch.where(
        inds,
        input
        * torch.reciprocal(
            torch.sqrt(cutoff.to(dtype=input.dtype, device=input.device))
        ),
        torch.sqrt(torch.where(inds, 1, input)),
    )
    return output


def robust_log(input: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    input = torch.abs(input)
    inds = input < cutoff
    output = torch.where(
        inds,
        math.log(cutoff) + (input / cutoff) - 1.0,
        torch.log(torch.where(inds, 1, input)),
    )
    return output


# NOTE: We wrote stable reciprocal in order to handle
# undefined points (f(0) = inf in this case),
# futher testing should be done.
def stable_reciprocal(input: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    inds = torch.abs(input) < cutoff
    y_c = torch.reciprocal(cutoff)
    output = torch.where(
        inds,
        (torch.sign(input) + (1 - torch.sign(torch.abs(input)))) * 2 * y_c
        + (-input / (cutoff**2)),
        torch.reciprocal(torch.where(inds, 1, input)),
    )
    return output


# Non linearity funcs
def sin(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(input)


def cos(input: torch.Tensor) -> torch.Tensor:
    return torch.cos(input)


def relu(input: torch.Tensor) -> torch.Tensor:
    return torch.relu(input)


def gelu(input: torch.Tensor) -> torch.Tensor:
    return F.gelu(input)


def leaky_relu(input: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    # TODO: Consider changing slope type
    return F.leaky_relu(input, slope)  # type: ignore


def tanh(input: torch.Tensor) -> torch.Tensor:
    return torch.tanh(input)


def sigmoid(input: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(input)


def softplus(input: torch.Tensor) -> torch.Tensor:
    return F.softplus(input)


def softmax(input: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    return F.softmax(input, dim=axis)


# Reduction ops
def reduce_mean(
    input: torch.Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.mean(input, dim=axis, keepdim=keepdim)


def reduce_sum(
    input: torch.Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.sum(input, dim=axis, keepdim=keepdim)


def reduce_max(
    input: torch.Tensor,
    *,
    axis: int | tuple[int, ...],
    keepdim: bool = False,
) -> torch.Tensor:
    # NOTE: torch.amax evenly distributes gradient between equal values!
    return torch.amax(input, dim=axis, keepdim=keepdim)


def reduce_argmax(
    input: torch.Tensor,
    *,
    axis: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.argmax(input, dim=axis, keepdim=keepdim)


def reduce_min(
    input: torch.Tensor,
    *,
    axis: int | tuple[int, ...],
    keepdim: bool = False,
) -> torch.Tensor:
    # NOTE: torch.amin evenly distributes gradient between equal values!
    return torch.amin(input, dim=axis, keepdim=keepdim)


def reduce_argmin(
    input: torch.Tensor,
    *,
    axis: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.argmin(input, dim=axis, keepdim=keepdim)


def reduce_prod(
    input: torch.Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    match axis:
        case None:
            if not keepdim:
                return torch.prod(input)
            else:
                for _ in range(len(input.shape)):
                    input = torch.unsqueeze(torch.prod(input, dim=0), -1)
                return input
        case int(axis):
            return torch.prod(input, dim=axis, keepdim=keepdim)
        case tuple(axis):
            if len(set(axis)) != len(axis):
                raise ValueError("Duplicate value in 'axis'")
            for idx, dim in enumerate(sorted(axis)):
                if not keepdim:
                    dim -= idx
                input = torch.prod(input, dim=dim, keepdim=keepdim)
            return input


def variance(
    input: torch.Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    correction: float = 0.0,
) -> torch.Tensor:
    return torch.var(input, dim=axis, correction=correction, keepdim=keepdim)


# NN ops
def conv1d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
) -> torch.Tensor:
    if isinstance(padding, Sequence):
        # TODO: padding should not be always tuple
        input = F.pad(input, [padding[1], padding[0]], "constant", 0)

    return torch.nn.functional.conv1d(
        input=input,
        weight=kernel,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=1,
    )


def conv1d_bias(
    input: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    if isinstance(padding, Sequence):
        input = F.pad(input, [padding[0], padding[1]], "constant", 0)

    return torch.nn.functional.conv1d(
        input=input,
        weight=kernel,
        bias=torch.atleast_1d(bias.squeeze()),
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=1,
    )


def conv2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    _padding: tuple[int, int]
    if is_int_tuple_tuple(padding):
        input = F.pad(input, [*padding[1], *padding[0]], "constant", 0)
        _padding = (0, 0)
    else:
        _padding = padding  # type: ignore

    # TODO: padding type will be fix with typeIs
    return F.conv2d(
        input=input,
        weight=kernel,
        bias=None,
        stride=stride,
        padding=_padding,
        dilation=dilation,
        groups=1,
    )


def conv2d_bias(
    input: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    _padding: tuple[int, int]
    if is_int_tuple_tuple(padding):
        input = F.pad(input, [*padding[1], *padding[0]], "constant", 0)
        _padding = (0, 0)
    else:
        _padding = padding  # type: ignore

    # TODO: padding type will be fix with typeIs
    return F.conv2d(
        input=input,
        weight=kernel,
        bias=torch.atleast_1d(bias.squeeze()),
        stride=stride,
        padding=_padding,
        dilation=dilation,
        groups=1,
    )


def max_pool1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int,
    *,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
) -> torch.Tensor:
    if isinstance(padding, Sequence):
        input = F.pad(input, [padding[0], padding[1]], "constant", 0)

    return F.max_pool1d(
        input,
        kernel_size,
        stride=stride,
        padding=0,
        dilation=dilation,
        ceil_mode=False,
        return_indices=False,
    )


def max_pool2d(
    input: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    *,
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
) -> torch.Tensor:
    _padding: tuple[int, int]
    if is_int_tuple_tuple(padding):
        input = F.pad(input, [*padding[1], *padding[0]], "constant", 0)
        _padding = (0, 0)
    else:
        _padding = padding  # type: ignore

    return F.max_pool2d(
        input,
        kernel_size,
        stride=stride,
        padding=_padding,
        dilation=dilation,
        ceil_mode=False,
        return_indices=False,
    )


def positional_encoding(input: torch.Tensor, hidden_dim: int, max_len: int):
    # TODO: Make positional encoding a composite model.
    pe = torch.zeros((max_len, hidden_dim))
    position = torch.arange(0, max_len)[:, None]
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2)
        * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe[None]
    return input + pe[:, : input.shape[1]]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | int | None = None,
):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


# Loss funcs
def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weights: list[float] | bool,
    cutoff: torch.Tensor,
    *,
    categorical: bool = True,
    robust: bool = False,
) -> torch.Tensor:
    log: partial | Callable = (
        partial(robust_log, cutoff=cutoff) if robust else torch.log
    )
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )

    if categorical:
        if torch.is_floating_point(target) or torch.is_complex(target):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )

        return (
            -log(torch.take_along_dim(input, target[:, None].long(), dim=1)[:, 0])
            * _weights[target]
        )
    return -torch.sum(target * log(input) * _weights, dim=1)


def cross_entropy_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    weights: list[float] | bool,
    cutoff: torch.Tensor,
    *,
    categorical: bool = True,
    robust: bool = False,
) -> torch.Tensor:
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    if not robust:
        shape = tuple(target.shape)
        if not categorical:
            shape = shape[0:1] + shape[2:]
        if target.shape != input.shape:
            if torch.is_floating_point(target) or torch.is_complex(target):
                raise ValueError(
                    f"Cross entropy got unexpected type for target '{target.dtype}'."
                )
            target = target.long()

        return F.cross_entropy(
            input, target, weight=_weights.squeeze(), reduction="none"
        ).reshape(shape)
    else:
        log = partial(robust_log, cutoff=cutoff) if robust else torch.log
        if categorical:
            if torch.is_floating_point(target) or torch.is_complex(target):
                raise ValueError(
                    f"Cross entropy got unexpected type for target '{target.dtype}'."
                )

            logits_max = torch.max(input, dim=1, keepdim=True)[0]
            with torch.no_grad():
                input -= logits_max
            label_logits = torch.take_along_dim(input, target[:, None].long(), dim=1)[
                :, 0
            ]
            log_normalizers = log(torch.sum(torch.exp(input), dim=1))
            return (log_normalizers - label_logits) * _weights[target]
        return -torch.sum(
            target * log_softmax(input, log, robust, axis=1) * _weights, dim=1
        )


def cross_entropy_with_log_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weights: list[float] | bool,
    *,
    categorical: bool = True,
) -> torch.Tensor:
    _weights = calculate_cross_entropy_class_weights(
        input, target, categorical, weights
    )
    if categorical:
        if torch.is_floating_point(target) or torch.is_complex(target):
            raise ValueError(
                f"Cross entropy got unexpected type for target '{target.dtype}'."
            )
        return (
            -torch.take_along_dim(input, target.long()[:, None], dim=1)[:, 0]
            * _weights[target]
        )
    return -torch.sum(target * input * _weights, dim=1)


def binary_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    cutoff: torch.Tensor,
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
) -> torch.Tensor:
    log: partial | Callable = (
        partial(robust_log, cutoff=cutoff) if robust else torch.log
    )
    # TODO: Use F.binary_cross_entropy
    if isinstance(pos_weight, bool) and pos_weight:
        _pos_weight = calculate_binary_class_weight(target)
    else:
        _pos_weight = pos_weight

    return -_pos_weight * target * log(input) - (1 - target) * log(1 - input)


def binary_cross_entropy_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    cutoff: torch.Tensor,
    *,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
) -> torch.Tensor:
    log: partial | Callable = (
        partial(robust_log, cutoff=cutoff) if robust else torch.log
    )
    _pos_weight: torch.Tensor | None
    if isinstance(pos_weight, bool):
        _pos_weight = (
            calculate_binary_class_weight(torch.sigmoid(target)) if pos_weight else None
        )
    elif pos_weight == 1.0:
        _pos_weight = None
    else:
        _pos_weight = torch.tensor(pos_weight)

    if not robust:
        return F.binary_cross_entropy_with_logits(
            input, target, reduction="none", pos_weight=_pos_weight
        )

    if _pos_weight is not None:
        log_weight = (_pos_weight - 1) * (target) + 1
        loss = (1 - target) * input - (log_weight * log_sigmoid(input, log, robust))
    else:
        loss = (1 - target) * input - log_sigmoid(input, log, robust)

    return loss


def hinge_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.maximum(
        torch.tensor([0.0], dtype=input.dtype, device=input.device),
        1.0 - target * input,
    )


def quad_hinge_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return hinge_loss(input, target) ** 2


def absolute_error(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(input, target, reduction="none")


def quantile_loss(
    input: torch.Tensor, target: torch.Tensor, quantile: torch.Tensor
) -> torch.Tensor:
    error = target - input
    return torch.maximum(quantile * error, (quantile - 1) * error)


def kl_divergence(
    input: torch.Tensor, target: torch.Tensor, cutoff: torch.Tensor
) -> torch.Tensor:
    # NOTE: Torch built-in kk-divergence expects log-probabilities for predictions.
    # So we provide stable log of input1 to the kl_div.
    # return F.kl_div(RobustLog.formula(input1), input2, reduction = "none")
    return F.kl_div(
        robust_log(input, cutoff),
        robust_log(target, cutoff),
        reduction="none",
        log_target=True,
    )


def eye(N: int, M: int, *, device: str, precision: int) -> torch.Tensor:
    if M is None:
        return handle_data_precision(torch.eye(N, device=device), precision)
    else:
        return handle_data_precision(torch.eye(N, M, device=device), precision)


def transposed_diag(input: torch.Tensor) -> torch.Tensor:
    return torch.diag(input)[:, None]


def ones_with_zero_diag(N: int, M: int, device: str, precision: int) -> torch.Tensor:
    if M is None:
        output = torch.ones(N) - torch.eye(N)
    else:
        output = torch.ones((N, M)) - torch.eye(N, M)
    return handle_data_precision(output, precision)


def unique(input: torch.Tensor) -> torch.Tensor:
    return torch.unique(input)


def trapezoid(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(y, x)


def primitive_accuracy(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    prediction = torch.argmax(input1, dim=1).reshape(input1.shape[0], 1)
    return torch.mean(prediction == input2)


def auc_core(input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    if input.ndim > 1:
        raise ValueError(f"Input should be 1D array, but given '{input.ndim}D'")
    if label.ndim > 1:
        raise ValueError(f"Label should be 1D array, but given '{label.ndim}D'")
    s_input, _ = torch.sort(torch.unique(input))
    tprs, fprs = torch.vmap(
        lambda threshold: calculate_tpr_fpr(threshold, input, label)
    )(torch.flip(s_input, dims=(0,)))

    return torch.stack((tprs, fprs))


def squeeze(input: torch.Tensor) -> torch.Tensor:
    return torch.squeeze(input)


def broadcast_to(input: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.broadcast_to(input, shape)


def transpose(
    input: torch.Tensor, axes: list[int] | tuple[int, ...] | None = None
) -> torch.Tensor:
    if axes is None:
        return input.permute(*reversed(range(input.ndim)))
    else:
        return input.permute(*axes)


def where(
    cond: torch.Tensor, input1: torch.Tensor, input2: torch.Tensor
) -> torch.Tensor:
    return torch.where(cond, input1, input2)


def to_tensor(
    input: NestedFloatOrIntOrBoolList, device: str, precision: int
) -> torch.Tensor:
    return torch.tensor(input, device=device, dtype=get_type(input, precision))


def tensor_to_list(input: torch.Tensor) -> NestedFloatOrIntOrBoolList:
    return input.tolist()


def to_parallel(tensor: torch.Tensor, device_mesh: DeviceMesh) -> torch.Tensor:
    return distribute_tensor(
        tensor, device_mesh, [Replicate()] * len(device_mesh.shape)
    )


def arange(*args, device: torch.device, precision: int) -> torch.Tensor:
    return handle_data_precision(torch.arange(*args, device=device), precision)


def concat(*inputs: torch.Tensor, axis: int | None = 0) -> torch.Tensor:
    if axis is None:
        return torch.concatenate([torch.flatten(v) for v in inputs])
    else:
        return torch.concatenate([torch.atleast_1d(v) for v in inputs], dim=axis)


def matrix_concat(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    return torch.concatenate((input1, input2), dim=input1.ndim - 1)


def stop_gradient(input: torch.Tensor) -> torch.Tensor:
    return input.detach()


def flatten(input: torch.Tensor, *, start_dim: int = 0, end_dim: int = 1):
    return torch.flatten(input, start_dim, end_dim)


def shape(input: torch.Tensor) -> tuple[int, ...]:
    return tuple(input.shape)


def size(
    input: torch.Tensor, dim: int | tuple[int, ...] | None
) -> int | tuple[int, ...]:
    if dim is None:
        return math.prod(input.size())
    if isinstance(dim, int):
        return input.size(dim)
    if isinstance(dim, Sequence):
        return tuple(input.size(idx) for idx in dim)
    else:
        raise ValueError(f"Unexpected dim: {dim}")


def norm_modifier(input: torch.Tensor) -> torch.Tensor:
    inner_term = ((input - 1.0) % 8) / 8 - 0.5
    return 4.0 * (1.0 - 2.0 * torch.abs(inner_term)) + 1.0


def lstm_cell(
    input: torch.Tensor,
    prev_cell: torch.Tensor,
    prev_hidden: torch.Tensor,
    w_i: torch.Tensor,
    w_f: torch.Tensor,
    w_c: torch.Tensor,
    w_o: torch.Tensor,
    bias_i: torch.Tensor,
    bias_f: torch.Tensor,
    bias_c: torch.Tensor,
    bias_o: torch.Tensor,
) -> torch.Tensor:
    input = input.squeeze(dim=1)
    cell = prev_cell.squeeze(dim=1)
    hidden = prev_hidden.squeeze(dim=1)
    input_features = input.shape[-1]

    w_ig = w_c[:input_features].T
    w_hg = w_c[input_features:].T

    w_if = w_f[:input_features].T
    w_hf = w_f[input_features:].T

    w_ii = w_i[:input_features].T
    w_hi = w_i[input_features:].T

    w_io = w_o[:input_features].T
    w_ho = w_o[input_features:].T

    weight_ih_l0 = torch.concat((w_ii, w_if, w_ig, w_io), dim=0)
    weight_hh_l0 = torch.concat((w_hi, w_hf, w_hg, w_ho), dim=0)
    bias_ih_l0 = torch.concat((bias_i, bias_f, bias_c, bias_o), dim=0)

    hidden_out, cell_out = torch._C._VariableFunctions.lstm_cell(
        input, (hidden, cell), weight_ih_l0, weight_hh_l0, bias_ih_l0
    )

    # input = input.squeeze(dim = 1)
    # cell = prev_cell.squeeze(dim = 1)
    # hidden = prev_hidden.squeeze(dim = 1)
    # hidden_shape = hidden.shape[-1]
    # input_hidden = torch.concat((input, hidden), dim = 1)
    # all_weights = torch.concat((w_i, w_f, w_c, w_o), dim = 1)
    # a_t = input_hidden @ all_weights
    # i_t = torch.sigmoid(a_t[:, 0: hidden_shape])
    # f_t = torch.sigmoid(a_t[:, hidden_shape: 2 * hidden_shape])
    # o_t = torch.sigmoid(a_t[:, 2 * hidden_shape: 3 * hidden_shape])
    # g_t = torch.tanh(a_t[:, 3 * hidden_shape: 4 * hidden_shape])
    # cell_out = f_t * cell + i_t * g_t
    # hidden_out = o_t * torch.tanh(cell_out)
    output = torch.concat((cell_out[:, None, :], hidden_out[:, None, :]), dim=0)
    return output


def distance_matrix(
    left: torch.Tensor, right: torch.Tensor, norm: torch.Tensor
) -> torch.Tensor:
    diffs = left[:, None, :] - right[None, :, :]
    abs_diffs = torch.abs(diffs)
    powered_abs_diffs = abs_diffs**norm
    powered_dists = torch.sum(powered_abs_diffs, 2)
    return powered_dists


def polynomial_features(input: torch.Tensor, *, degree: int = 2) -> torch.Tensor:
    samples, dims = input.shape
    identity = torch.eye(dims + 1, dims + 1, dtype=torch.int)
    data = torch.hstack(
        (
            torch.ones((samples, 1), device=input.device, dtype=input.dtype),
            input,
        )
    )
    powers: Iterator = map(sum, combinations_with_replacement(identity, degree))
    # Skip first element of powers. This is the bias term.
    next(powers)
    return torch.hstack([(data**p).prod(1)[:, None] for p in powers])


def tsne_p_joint(
    squared_distances: torch.Tensor,
    target_perplexity: torch.Tensor,
    threshold: torch.Tensor,
) -> torch.Tensor:
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
    # Get the negative euclidian distances matrix for our data
    negative_dist_sq = (-1) * squared_distances
    # Find optimal sigma for each row of this distances matrix
    # TODO: Fix wrong types
    sigmas = find_optimal_sigmas(negative_dist_sq, target_perplexity, threshold)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(negative_dist_sq, sigmas)
    # Go from conditional to joint probabilities matrix in a symmetrical manner.
    P = (p_conditional + p_conditional.T) / (2.0 * p_conditional.shape[0])
    return P


def cholesky(input: torch.Tensor) -> torch.Tensor | None:
    try:
        return torch.linalg.cholesky(input)
    except torch.linalg.LinAlgError as e:
        logging.info(str(e))
    return None


def gpr_alpha(
    label_mu_diff: torch.Tensor, L: torch.Tensor, K_term: torch.Tensor
) -> torch.Tensor:
    if L is not None:
        alpha = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, label_mu_diff, lower=True),
            lower=False,
        )
    else:
        alpha = torch.linalg.solve(K_term, label_mu_diff)
    return alpha


def eigvalsh(
    K_term: torch.Tensor, L: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    if L is not None:
        return torch.diag(L)
    else:
        return torch.clip(torch.linalg.eigvalsh(K_term), threshold, None) / 2


def gpr_v_outer(K: torch.Tensor, K_term: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    if L is not None:
        v = torch.linalg.solve_triangular(L, K, lower=True)
        v_outer = v.T @ v
    else:
        v_outer = K.T @ torch.linalg.lstsq(K_term, K)[0]
    return v_outer


def isnan(input: torch.Tensor):
    return torch.isnan(input)


def nan_to_num(
    input: torch.Tensor,
    nan: int | float | None,
    posinf: int | float | None,
    neginf: int | float | None,
):
    return torch.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)


def astype(input: torch.Tensor, dtype: core.Dtype | int) -> torch.Tensor:
    return handle_data_dtype(input, dtype)


def dtype(input: torch.Tensor) -> core.Dtype:
    return getattr(core, str(input.dtype).split(".")[-1])


def logical_xor(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.logical_xor(left, right)


def split(input: torch.Tensor, split_size: int | list[int], axis: int = 0):
    return torch.stack(torch.tensor_split(input, split_size, dim=axis))


def pad(input: torch.Tensor, pad_width: tuple[tuple[int, int], ...]):
    _padding = tuple(pad_item for pad in reversed(pad_width) for pad_item in pad)
    return F.pad(input, _padding, "constant", 0)


array_creation_funcs = ["arange", "to_tensor", "eye", "ones_with_zero_diag"]
primitive_func_dict = common_primitive_func_dict | {
    key: fn for key, fn in globals().items() if callable(fn)
}
