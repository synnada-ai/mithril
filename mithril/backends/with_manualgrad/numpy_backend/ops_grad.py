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

import itertools
from itertools import zip_longest

import numpy as np
import scipy.linalg as slin  # type: ignore[import-untyped]
from scipy.special import erf  # type: ignore[import-untyped]

from ....utils.type_utils import is_tuple_int
from .ops import hinge_loss, sigmoid, softmax
from .utils import (
    CacheType,
    accumulate_grads,
    calc_input_slices,
    get_submatrices1d,
    get_submatrices2d,
    verify_shapes,
    write_into_cache,
)

__all__ = [
    "accumulate_grads",
    "write_into_cache",
    "matrix_multiplication_grad",
    "multiplication_grad",
    "divide_grad",
    "add_grad",
    "subtract_grad",
    "squared_error_grad",
    "absolute_error_grad",
    "cross_entropy_grad",
    "cross_entropy_with_logits_grad",
    "cross_entropy_with_log_probs_grad",
    "binary_cross_entropy_grad",
    "binary_cross_entropy_with_logits_grad",
    "quantile_loss_grad",
    "hinge_loss_grad",
    "quad_hinge_loss_grad",
    "kl_divergence_grad",
    "power_grad",
    "exp_grad",
    "sqrt_grad",
    "sin_grad",
    "cos_grad",
    "robust_sqrt_grad",
    "robust_log_grad",
    "stable_reciprocal_grad",
    "cartesian_diff_grad",
    "abs_grad",
    "sign_grad",
    "concat_grad",
    "reduce_mean_grad",
    "reduce_sum_grad",
    "reduce_max_grad",
    "reduce_min_grad",
    "reduce_prod_grad",
    "buffer_grad",
    "relu_grad",
    "leaky_relu_grad",
    "tanh_grad",
    "sigmoid_grad",
    "softplus_grad",
    "gelu_grad",
    "stop_gradient_grad",
    "tensor_slice_grad",
    "tensor_item_grad",
    "permute_tensor_grad",
    "transpose_grad",
    "square_grad",
    "conv1d_grad",
    "conv1d_bias_grad",
    "conv2d_grad",
    "conv2d_bias_grad",
    "where_grad",
    "nan_to_num_grad",
    "isnan_grad",
    "scaled_dot_product_attention_grad",
    "primitive_embedding_grad",
    "reshape_grad",
    "variance_grad",
    "swapaxes_grad",
    "broadcast_to_grad",
    "squeeze_grad",
    "log_grad",
    "transposed_diag_grad",
    "gpr_v_outer_grad",
    "eigvalsh_grad",
    "gpr_alpha_grad",
    "cholesky_grad",
    "polynomial_features_grad",
    "distance_matrix_grad",
    "norm_modifier_grad",
    "robust_power_grad",
    "softmax_grad",
    "max_pool2d_grad",
    "max_pool1d_grad",
    "flatten_grad",
    "minus_grad",
]


def matrix_multiplication_grad(
    output_gradient: np.ndarray,
    cache: CacheType | None,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    if idx == 0:
        return output_gradient @ np.swapaxes(inputs[1], -1, -2)
    elif idx == 1:
        return np.swapaxes(inputs[0], -1, -2) @ output_gradient
    else:
        raise ValueError("Invalid index for matrix multiplication gradient.")


def multiplication_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    input1, input2 = inputs
    return [input2, input1][idx] * output_gradient


def divide_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:  #  type: ignore
    verify_shapes(inputs, idx)
    input1, input2 = inputs
    if idx == 0:
        return output_gradient / input2
    elif idx == 1:
        return -output_gradient * input1 / (input2**2)
    else:
        raise ValueError("Invalid index for divide gradient.")


def add_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return output_gradient


def subtract_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return [1, -1][idx] * output_gradient


def squared_error_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    input, target = inputs
    grad = 2 * (input - target) * output_gradient
    return [1, -1][idx] * grad


def absolute_error_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    signs = np.sign(cache["diff"]) * output_gradient
    return [1, -1][idx] * signs


def cross_entropy_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    categorical: bool = True,
    robust: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1, 2])
    input, target, _, cutoff = inputs
    if categorical:
        grad = np.zeros_like(input)
        np.put_along_axis(grad, target[:, None], 1, axis=1)
        output_gradient *= cache["weights"][target]
        if robust:
            grad *= robust_log_grad(
                output_gradient,
                cache,
                0,
                np.take_along_axis(input, target[:, None], axis=1)[:, 0, ...],
                cutoff,
            )[:, None, ...]
        else:
            grad *= log_grad(
                output_gradient,
                cache,
                0,
                np.take_along_axis(input, target[:, None], axis=1)[:, 0, ...],
            )[:, None]
        return -grad

    s_grad = reduce_sum_grad(output_gradient, None, 0, input, axis=1, keepdim=False)
    s_grad = s_grad * cache["weights"]
    if not robust:
        l_grad = log_grad(s_grad, None, 0, input) * target
    else:
        l_grad = robust_log_grad(s_grad, None, 0, input, cutoff) * target
    return -l_grad


def cross_entropy_with_logits_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    categorical: bool = True,
    robust: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, target, _, cutoff = inputs
    grad = softmax(input, axis=1)
    if categorical:
        grad_c = np.zeros_like(grad)
        np.put_along_axis(grad_c, target[:, None], 1, axis=1)
        grad -= grad_c
        return grad * (output_gradient * cache["weights"][target])[:, None]

    # TODO: This grad algorthim inefficient make it better
    s_grad = reduce_sum_grad(output_gradient, None, 0, input, axis=1, keepdim=False)
    s_grad = s_grad * cache["weights"]
    if not robust:
        l_grad = log_grad(s_grad, None, 0, grad) * target
    else:
        l_grad = robust_log_grad(s_grad, None, 0, grad, cutoff) * target

    return -softmax_grad(l_grad, {"output": grad, "axis": 1}, 0, input)


def cross_entropy_with_log_probs_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    categorical: bool = True,
) -> np.ndarray:
    input, target, _ = inputs

    if categorical:
        verify_shapes(inputs, idx, non_differentiables=[1, 2])
        grad = np.zeros_like(input)
        np.put_along_axis(grad, target[:, None], 1, axis=1)
        return -grad * (output_gradient * cache["weights"][target])[:, None]

    return (
        -reduce_sum_grad(output_gradient, None, 0, input, axis=1, keepdim=False)
        * target
        * cache["weights"]
    )


def binary_cross_entropy_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1, 2])
    (
        input,
        target,
        cutoff,
        *_,
    ) = inputs
    pos_weight = cache["pos_weight"]
    if robust:
        grad = -pos_weight * target * robust_log_grad(
            output_gradient, cache, 0, input, cutoff
        ) + (1 - target) * robust_log_grad(output_gradient, cache, 0, 1 - input, cutoff)
    else:
        grad = -pos_weight * target * output_gradient * (1 / input) + (
            1 - target
        ) * output_gradient * (1 / (1 - input))
    return grad


def binary_cross_entropy_with_logits_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    pos_weight: bool | float = 1.0,
    robust: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, target, *_ = inputs

    if "pos_weight" in cache:
        _pos_weight = cache["pos_weight"]
        _pos_weight *= target

        return (
            (_pos_weight + 1 - target) * sigmoid(input) - _pos_weight
        ) * output_gradient

    else:
        return (sigmoid(input) - target) * output_gradient


def quantile_loss_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[2])
    input, target, quantile = inputs
    dloss = np.sign(target - input)
    dloss[dloss == 0] = quantile - 0.5
    dloss[dloss == 1] = quantile
    dloss[dloss == -1] = quantile - 1.0
    # TODO: Discuss
    return [-1, 1][idx] * dloss * output_gradient


def hinge_loss_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])

    # TODO: Discuss -> Since we may return original output to the user, \
    #  don't modify it in-place.
    out_copy = cache["output"].copy()
    out_copy[out_copy > 0] = 1.0
    out_copy[cache["base_hinge"] == 0.0] = 0.5
    return -np.multiply(out_copy, inputs[1]) * output_gradient


def quad_hinge_loss_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, target = inputs
    return (
        -2.0 * np.multiply(hinge_loss(input, target, {}), inputs[1]) * output_gradient
    )


def kl_divergence_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[2])
    input, target, cutoff = inputs
    grad = (
        [-1, 1][idx] * target * robust_log_grad(output_gradient, {}, 0, input, cutoff)
    )
    if idx == 1:
        grad += cache["partial_result"]
    return grad


def power_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    input, target = inputs
    if idx == 0:
        grad = target * (input ** (target - 1.0)) * output_gradient
    elif idx == 1:
        grad = (input**target) * np.log(input) * output_gradient
        # TODO: Handle the situation when input = 0.0.
        grad *= input != 0.0
        # TODO: Needs to be performed more efficiently.
        #  We're first calculating nan values and correcting after that!
    return grad


def exp_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return cache["output"] * output_gradient


def sqrt_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return (1 / 2) * (1 / cache["output"]) * output_gradient


def sin_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return np.cos(inputs[0]) * output_gradient


def cos_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return -np.sin(inputs[0]) * output_gradient


def robust_sqrt_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, cutoff = inputs
    inds = np.abs(input) < cutoff
    grad = np.zeros_like(input)
    grad[~inds] = (1 / 2) * (1 / cache["output"][~inds])
    grad[inds] = np.reciprocal(np.sqrt(cutoff))
    return np.sign(input) * grad * output_gradient


# NOTE: We wrote the stabilized log in order to avoid
# undefined points (log(0) = -inf in this case),
# further testing should be done about performance.
def robust_log_grad(
    output_gradient: np.ndarray,
    cache: CacheType | None,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, cutoff = inputs
    negative_inds = input < 0.0
    input = np.abs(input)
    inds = input < cutoff
    grad = np.zeros_like(input)
    grad[~inds] = 1 / input[~inds]
    grad[inds] = 1 / cutoff
    grad[negative_inds] = -grad[negative_inds]
    # grad[positive_inds] = grad[positive_inds]
    return grad * output_gradient


# NOTE: We wrote stable reciprocal in order to handle
# undefined points (f(0) = inf in this case),
# futher testing should be done.
def stable_reciprocal_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, cutoff = inputs
    inds = np.abs(input) < cutoff
    grad = np.zeros_like(input)
    grad[~inds] = -1 / np.square(input[~inds])
    grad[inds] = -(1 / np.square(cutoff))
    return grad * output_gradient


def cartesian_diff_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return [1, -1][idx] * np.sum(output_gradient, axis=1 - idx)


def abs_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    # sign[sign == 0] = 1 # NOTE: JAX returns 1.0 for gradient at this point!!!
    return np.sign(inputs[0]) * output_gradient


def sign_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return np.zeros_like(inputs[0])


def concat_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    axis=0,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[-1])
    # Since last element of args is axis, exclude it from
    # gradient calculation.
    slices = write_into_cache(
        cache,
        "slices",
        (output_gradient, axis, *inputs),
        constant=True,
        func=calc_input_slices,
    )
    key_slice = slices[f"input{idx + 1}"]  # type: ignore
    if axis is not None:
        return output_gradient[key_slice]
    else:
        return output_gradient[key_slice].reshape(inputs[idx].shape)


def reduce_mean_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    # TODO: Update gradient formula (same as Sum).
    (input,) = inputs
    input_shape = input.shape
    if axis is None:
        size = input.size
        return output_gradient * np.ones(input_shape) / size
    else:
        if isinstance(axis, int):
            size = input_shape[axis]
        else:
            size = 1
            for idx in axis:
                size *= input_shape[idx]
        if not keepdim:
            output_gradient = np.expand_dims(output_gradient, axis=axis)
        return np.broadcast_to(output_gradient, input_shape) / size


def reduce_sum_grad(
    output_gradient: np.ndarray,
    cache: CacheType | None,
    idx: int,
    *inputs: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input_shape = inputs[0].shape
    if axis is None:
        return output_gradient * np.ones(input_shape)
    else:
        if not keepdim:
            output_gradient = np.expand_dims(output_gradient, axis=axis)
        return np.broadcast_to(output_gradient, input_shape)


def reduce_max_grad(
    output_gradient: np.ndarray,
    cache: CacheType | None,
    idx: int,
    *inputs: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    (input,) = inputs
    input_shape = input.shape
    # Expand dimensions of output gradient to match the input shape
    # Create a mask indicating the maximum values along the specified axis
    if axis is None:
        max_mask = input.max(keepdims=True) == input
        return max_mask * output_gradient / np.sum(max_mask)
    else:
        if not keepdim:
            output_gradient = np.expand_dims(output_gradient, axis=axis)
        output_gradient = np.broadcast_to(output_gradient, input_shape)
        max_mask = input.max(axis=axis, keepdims=True) == input
        return max_mask * output_gradient / np.sum(max_mask, axis=axis, keepdims=True)
    # Calculate the input gradient using the max mask and broadcasting
    # NOTE: Since JAX and Torch distributes output gradient evenly to
    # the all max values, we do so in Numpy too.


def reduce_min_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    (input,) = inputs
    # Expand dimensions of output gradient to match the input shape
    # Create a mask indicating the maximum values along the specified axis
    if axis is None:
        min_mask = input.min(keepdims=True) == input
        return min_mask * output_gradient / np.sum(min_mask)
    else:
        if not keepdim:
            output_gradient = np.expand_dims(output_gradient, axis=axis)
        output_gradient = np.broadcast_to(output_gradient, input.shape)
        min_mask = input.min(axis=axis, keepdims=True) == input
        return min_mask * output_gradient / np.sum(min_mask, axis=axis, keepdims=True)
    # Calculate the input gradient using the max mask and broadcasting
    # NOTE: Since JAX and Torch distributes output gradient evenly to
    # the all max values, we do so in Numpy too.


def reduce_prod_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> np.ndarray:
    (input,) = inputs

    _axis: tuple[int, ...]
    if axis is None:
        _axis = tuple(idx for idx in range(len(input.shape)))
    elif isinstance(axis, int):
        _axis = (axis,)
    else:
        assert is_tuple_int(axis)
        _axis = axis

    if not keepdim:
        output = np.expand_dims(cache["output"], _axis) / input
    else:
        output = cache["output"] / input

    if np.sum(cache["output"] == 0) > 0.0:
        zero_indexes = np.argwhere(input == 0.0)
        for index in zero_indexes:
            index_tuple = tuple(
                slice(None, None, None) if idx in _axis else num
                for idx, num in enumerate(index)
            )
            prod_array = np.ma.array(input[index_tuple], mask=False)
            prod_idx = tuple(num for idx, num in enumerate(index) if idx in _axis)
            prod_array.mask[prod_idx] = True
            output[tuple(index)] = prod_array.prod()
    if not keepdim:
        return output * np.expand_dims(output_gradient, _axis)
    else:
        return output * output_gradient


def buffer_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return output_gradient


def relu_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    (input,) = inputs
    # TODO: Check if copy is necessary.
    inp_copy = np.copy(input)
    inp_copy[inp_copy > 0.0] = 1.0
    inp_copy[inp_copy <= 0.0] = 0.0
    return output_gradient * inp_copy


def leaky_relu_grad(output_gradient, cache, idx, *inputs):
    verify_shapes(inputs, idx)
    if idx == 0:
        # TODO: Check if copy is necessary.
        inp_copy = np.copy(inputs[0])
        slope = inputs[1]
        inp_copy[inp_copy > 0.0] = 1.0
        inp_copy[inp_copy <= 0.0] = slope
        return output_gradient * inp_copy
    elif idx == 1:
        return output_gradient * np.minimum(0.0, inputs[0])


def tanh_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return output_gradient * (1.0 - cache["output"] ** 2)


def sigmoid_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    sig = cache["output"]
    return output_gradient * (sig * (1 - sig))


def softplus_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    sig = sigmoid(inputs[0])
    return output_gradient * sig


def gelu_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    input, *_ = inputs
    s = input / np.sqrt(2)
    erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x**2))  # noqa: E731
    grad = 0.5 + 0.5 * erf(s) + ((0.5 * input * erf_prime(s)) / np.sqrt(2))
    return grad * output_gradient


def stop_gradient_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    return np.zeros_like(output_gradient)


# def tensor_slice_grad(output_gradient: np.ndarray,
#                       cache: CacheType,
#                       idx: int,
#                       *inputs: np.ndarray, -> np.ndarray:
#     verify_shapes(inputs, idx)
#     input1, input2 = inputs
#     if idx == 0:
#         grad = np.zeros_like(input1)
#         grad[:input2.shape[0], ...] = output_gradient
#         return grad
#     elif idx == 1:
#         return np.zeros_like(input2)


def tensor_slice_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1, 2, 3, 4])
    input, start, stop, step = inputs
    grad = np.zeros_like(input)
    grad[start:stop:step] = output_gradient
    return grad


def tensor_item_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, index = inputs
    grad = np.zeros_like(input)
    grad[index] = output_gradient
    return grad


def permute_tensor_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    indices = inputs[1]
    return output_gradient[np.argsort(indices)]


def transpose_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    axes = inputs[1]
    return (
        output_gradient.transpose(*np.argsort(axes))
        if axes is not None
        else output_gradient.T
    )


def square_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return output_gradient * 2 * inputs[0]


def conv1d_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[2, 3, 4])
    input1, input2 = inputs
    n, c, w = input1.shape
    _, _, w_k = input2.shape
    out_w = (w - w_k + sum(padding)) // stride + 1
    if idx == 0:
        _output_gradient = np.zeros(
            (
                output_gradient.shape[0],
                output_gradient.shape[1],
                (output_gradient.shape[2] - 1) * stride + 1,
            ),
            dtype=output_gradient.dtype,
        )
        _output_gradient[:, :, ::stride] = output_gradient
        _output_gradient = np.pad(
            _output_gradient,
            pad_width=((0,), (0,), (w_k - 1,)),
            mode="constant",
            constant_values=(0.0,),
        )
        submatrices_in1 = get_submatrices1d(
            _output_gradient,
            (n, c, _output_gradient.shape[2] - (w_k // 2) * 2),
            w_k,
            (0, 0),
            1,
        )
        rot_kernel = np.flip(input2, 2)
        grad = np.einsum("nowl,oil->niw", submatrices_in1, rot_kernel)
        input_grad = np.zeros_like(input1)
        grad = grad[:, :, padding[0] : padding[0] + w]
        input_grad[:, :, : grad.shape[2]] = grad
        return input_grad
    elif idx == 1:
        submatrices_in2 = get_submatrices1d(input1, (n, c, out_w), w_k, padding, stride)
        return np.einsum("niwl,now->oil", submatrices_in2, output_gradient)

    else:
        raise ValueError("Invalid index for conv1d gradient.")


def conv1d_bias_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    stride: int = 1,
    padding: tuple[int, int] = (1, 1),
    dilation: int = 1,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[3, 4, 5])
    if idx < 2:
        return conv1d_grad(
            output_gradient,
            cache,
            idx,
            *inputs[:2],
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    elif idx == 2:
        return output_gradient.sum((0,), keepdims=True)
    else:
        raise ValueError("Invalid index for conv1d bias gradient.")


def conv2d_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[2, 3, 4])
    input1, input2 = inputs

    _padding: tuple[tuple[int, int], tuple[int, int]]
    if is_tuple_int(padding):
        _padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        _padding = padding  # type: ignore

    n, c, h, w = input1.shape
    _, _, h_k, w_k = input2.shape
    out_h = (h - h_k + sum(_padding[0])) // stride[0] + 1
    out_w = (w - w_k + sum(_padding[1])) // stride[1] + 1

    if idx == 0:
        _output_gradient = np.zeros(
            (
                output_gradient.shape[0],
                output_gradient.shape[1],
                (output_gradient.shape[2] - 1) * stride[0] + 1,
                (output_gradient.shape[3] - 1) * stride[1] + 1,
            ),
            dtype=output_gradient.dtype,
        )
        _output_gradient[:, :, :: stride[0], :: stride[1]] = output_gradient
        _output_gradient = np.pad(
            _output_gradient,
            pad_width=((0,), (0,), (h_k - 1,), (w_k - 1,)),
            mode="constant",
            constant_values=(0.0,),
        )
        submatrices_in1 = get_submatrices2d(
            _output_gradient,
            (
                n,
                c,
                _output_gradient.shape[2] - (h_k // 2) * 2,
                _output_gradient.shape[3] - (w_k // 2) * 2,
            ),
            h_k,
            w_k,
            ((0, 0), (0, 0)),
            1,
        )
        rot_kernel = np.rot90(input2, 2, axes=(2, 3))
        grad = np.einsum("nohwkl,oikl->nihw", submatrices_in1, rot_kernel)
        input_grad = np.zeros_like(input1)
        grad = grad[
            :,
            :,
            _padding[0][0] : _padding[0][0] + h,
            _padding[1][0] : _padding[1][0] + w,
        ]
        input_grad[:, :, : grad.shape[2], : grad.shape[3]] = grad
        return input_grad
    elif idx == 1:
        submatrices_in2 = get_submatrices2d(
            input1, (n, c, out_h, out_w), h_k, w_k, _padding, stride[0]
        )
        return np.einsum("nihwkl,nohw->oikl", submatrices_in2, output_gradient)

    else:
        raise ValueError("Invalid index for conv2d gradient.")


def conv2d_bias_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[3, 4, 5])
    if idx < 2:
        return conv2d_grad(
            output_gradient,
            cache,
            idx,
            *inputs[:2],
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    elif idx == 2:
        return output_gradient.sum((0,), keepdims=True)
    else:
        raise ValueError("Invalid index for conv2d bias gradient.")


def flatten_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    start_dim: int = 0,
    end_dim: int = -1,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1, 2])
    return output_gradient.reshape(*inputs[0].shape)


def max_pool1d_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    kernel_size: int,
    stride: int,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
) -> np.ndarray:
    if idx == 0:
        (input,) = inputs
        *_, w = input.shape
        out_w = (w - kernel_size + sum(padding)) // stride + 1
        padded_input = np.pad(
            input,
            pad_width=((0, 0), (0, 0), (padding[0], padding[1])),
            mode="constant",
            constant_values=(0.0,),
        )
        padded_shape = padded_input.shape
        dx = np.zeros_like(padded_input).astype(float)
        for j in range(out_w):
            start_w, end_w = j * stride, j * stride + kernel_size
            selected_window = padded_input[:, :, start_w:end_w]
            val = np.max(selected_window, axis=2)
            mask = val[:, :, None] == selected_window
            dx[:, :, start_w:end_w] += mask * (output_gradient[:, :, j])[:, :, None]

        return dx[..., padding[0] : padded_shape[-1] - padding[1]]

    else:
        raise ValueError("Invalid index for max_pool1d gradient.")


def max_pool2d_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    if idx == 0:
        (input,) = inputs

        normalized_padding: tuple[tuple[int, int], tuple[int, int]]
        if is_tuple_int(padding):
            normalized_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            normalized_padding = padding  # type: ignore

        if isinstance(stride, int):
            stride = (stride, stride)

        *_, h, w = input.shape
        h_k, w_k = kernel_size
        out_h = (h - kernel_size[0] + sum(normalized_padding[0])) // stride[0] + 1
        out_w = (w - kernel_size[1] + sum(normalized_padding[1])) // stride[1] + 1
        padded_input = np.pad(
            input,
            pad_width=(
                (0, 0),
                (0, 0),
                (normalized_padding[0][0], normalized_padding[0][1]),
                (normalized_padding[1][0], normalized_padding[1][1]),
            ),
            mode="constant",
            constant_values=(0.0,),
        )
        padded_shape = padded_input.shape
        dx = np.zeros_like(padded_input).astype(float)
        for i, j in itertools.product(range(out_h), range(out_w)):
            start_h, end_h = i * stride[0], i * stride[0] + h_k
            start_w, end_w = j * stride[1], j * stride[1] + w_k
            selected_window = padded_input[:, :, start_h:end_h, start_w:end_w]
            val = np.max(selected_window, axis=(2, 3))
            mask = val[:, :, None, None] == selected_window
            dx[:, :, start_h:end_h, start_w:end_w] += (
                mask * (output_gradient[:, :, i, j])[:, :, None, None]
            )
        return dx[
            ...,
            normalized_padding[0][0] : padded_shape[-2] - normalized_padding[0][1],
            normalized_padding[1][0] : padded_shape[-1] - normalized_padding[1][1],
        ]

    else:
        raise ValueError("Invalid index for max_pool2d gradient.")


def softmax_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    output = cache["output"]
    axis = cache["axis"]
    return output * (
        output_gradient - (output * output_gradient).sum(axis, keepdims=True)
    )


def robust_power_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[2])
    input1, input2, threshold = inputs
    result_shape = np.broadcast_shapes(input1.shape, input2.shape)
    input1 = np.broadcast_to(input1, result_shape)
    input1_sign = np.array(np.sign(input1))
    input1 = np.array(np.abs(input1))
    input2 = np.broadcast_to(input2, result_shape)
    threshold = np.broadcast_to(threshold, result_shape)
    if idx == 0:
        cond = cache["cond"]
        grad = cache.setdefault("in1_grad", np.zeros_like(cond, dtype=input1.dtype))
        if np.any(cond):
            grad[cond] = input1_sign[cond] * (
                (1 / threshold[cond]) * output_gradient[cond]
            )
        grad[~cond] = (
            input1_sign[~cond]
            * (input2[~cond] * (input1[~cond] ** (input2[~cond] - 1.0)))
            * output_gradient[~cond]
        )
    elif idx == 1:
        cond = cache["cond"]
        grad = cache.setdefault("in2_grad", np.zeros_like(cond, dtype=input1.dtype))
        if np.any(cond):
            grad[cond] = 0.0
        # Handle the situation when input1 == 0.0 where ln(input1) = -inf.

        grad[~cond] = (
            (input1[~cond] ** input2[~cond]) * np.log(input1[~cond])
        ) * output_gradient[~cond]
        grad[input1 == 0.0] = 0.0
    return grad


def norm_modifier_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    inner_term = cache["inner_term"]
    inner_term_item = inner_term if isinstance(inner_term, float) else inner_term.item()
    # TODO: Note that we don't have zero grad at minimum points
    # of triangular waveform while having at maximum points.
    # Should it be fixed??
    if inner_term != 0.0:
        if inner_term_item > 0.0:
            return -output_gradient
        return output_gradient
    return np.zeros_like(output_gradient)


def distance_matrix_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    norm = inputs[2]
    abs_diffs = cache["abs_diffs"]
    diffs_signs = np.sign(cache["diffs"])
    difference_grad = norm * (abs_diffs ** (norm - 1)) * diffs_signs
    if idx == 0:
        return np.einsum("ij, ijk -> ik", output_gradient, difference_grad)
    elif idx == 1:
        return np.einsum("ij, ijk -> jk", output_gradient, -difference_grad)
    elif idx == 2:
        powered_abs_diffs = cache["powered_abs_diffs"]
        # Handle "nan" terms resulting from log operation where
        # abs_diffs includes 0.0 terms.
        logs = np.nan_to_num(np.log(abs_diffs), posinf=0.0, neginf=0.0)
        terms = np.einsum("ijk -> ij", powered_abs_diffs * logs)
        return np.sum(output_gradient * terms).reshape(norm.shape)

    else:
        raise ValueError("Invalid index for distance_matrix gradient.")


def polynomial_features_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    degree: int = 2,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    samples, dims = inputs[0].shape
    powers = cache["powers"]
    data = cache["data"]
    zero_vec = np.zeros((samples, 1))  # TODO: Could this be cached?
    return np.einsum(
        "kij, ik->ij",
        np.array(
            [
                np.hstack(
                    [
                        p[d]
                        * (data ** [p_i - (i == d) for i, p_i in enumerate(p)]).prod(1)[
                            :, np.newaxis
                        ]
                        if p[d] > 0
                        else zero_vec
                        for d in range(1, dims + 1)
                    ]
                )
                for p in powers
            ]
        ),
        output_gradient,
    )


# def ones_with_zero_diag_grad(output_gradient: np.ndarray,
#                              cache: CacheType,
#                              idx: int,
#                              *inputs: np.ndarray, -> np.ndarray:
#     verify_shapes(inputs, idx)
#     return np.zeros_like(inputs[0])

# def eye_grad(output_gradient: np.ndarray,
#              cache: CacheType,
#              idx: int,
#              *inputs: np.ndarray, -> np.ndarray:
#     # TODO: Remove gradient formula!!!
#     return np.zeros_like(inputs[0])


def cholesky_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    # TODO: Implement cholesky gradient!
    raise NotImplementedError("Implement gradient of Cholesky Factorization!")


def gpr_alpha_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    label_mu_diff, L, K_term = inputs
    raise NotImplementedError()

    # if L is not None:
    #     l_inv = np.linalg.inv(L)
    #     diff_grad = l_inv.T @ l_inv @ output_gradient
    #     # (np.kron(l_inv.T, (l_inv.T @ l_inv @ label_mu_diff)) \
    #  @ output_gradients.reshape(-1,1)).reshape(2,2)
    #     # np.tril(-l_inv.T @ ((l_inv @ np.array(label_mu_diff)) \
    #  @ np.array([[3.0, 2]])) @ l_inv.T)
    # else:
    #     # IMPLEMENT
    #     return
    # return None


def eigvalsh_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[2])
    k_term, l_, threshold = inputs
    if idx == 0:
        if l_ is not None:
            return np.array(0.0, dtype=output_gradient.dtype)
        output = cache["output"]
        v = np.linalg.eig(k_term)[1]  # eigenvectors
        v_inv = np.linalg.inv(v)
        grad = np.zeros_like(k_term)
        for i in range(k_term.shape[0]):
            if output[i] > threshold:
                vi = v_inv[:, i]
                grad += np.outer(vi, vi) * output_gradient[i]
        return grad
    elif idx == 1:
        if l_ is not None:
            return np.eye(len(l_))
        return np.array(0.0, dtype=output_gradient.dtype)

    else:
        raise ValueError("Invalid index for eigvalsh gradient.")


def gpr_v_outer_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    k_, _, l_ = inputs
    if idx == 0:
        if l_ is not None:
            # TODO: Put v into a cache
            v = slin.solve_triangular(l_, k_, lower=True)
            v_grad = (output_gradient @ v.T).T + v @ output_gradient
            l_inv = np.linalg.inv(l_)
            # K_grad = (np.kron(l_inv.T, np.eye(2))
            # @ v_grad.reshape(-1,1)).reshape(2,2)
            k_grad = l_inv.T @ v_grad
            return k_grad
        else:
            # TODO: IMPLEMENT HERE
            raise NotImplementedError()
    elif idx == 1:
        if l_ is not None:
            return np.array(0.0, dtype=output_gradient.dtype)
        else:
            # TODO: IMPLEMENT HERE
            raise NotImplementedError()
    elif idx == 2:
        if l_ is not None:
            l_inv = np.linalg.inv(l_)
            # L_grad = (np.kron(l_inv.T,
            # (l_inv @ k_)) @ v_grad.reshape(-1,1)).reshape(2,2)
            l_grad = np.tril(-l_inv.T @ (output_gradient @ k_.T) @ l_inv.T)
            return l_grad
        else:
            # IMPLEMENT HERE
            raise NotImplementedError()

    else:
        raise ValueError("Invalid index for gpr_v_outer gradient.")


def transposed_diag_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    (input,) = inputs
    eye_mat = np.eye(input.shape[0])
    return output_gradient * eye_mat


def log_grad(
    output_gradient: np.ndarray,
    cache: CacheType | None,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return output_gradient * (1 / inputs[0])


def squeeze_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return output_gradient.reshape(inputs[0].shape)


def broadcast_to_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    input, shape = inputs[0], inputs[1]
    input_shape = input.shape
    bcast_indexes = []
    for idx, (in_s, out_s) in enumerate(zip_longest(input_shape[::-1], shape[::-1])):
        cur_axis = -idx - 1
        if in_s is None:
            bcast_indexes.append(cur_axis)
        else:
            if in_s != out_s:
                bcast_indexes.append(cur_axis)
    if bcast_indexes:
        in_grad = np.sum(output_gradient, axis=tuple(bcast_indexes))
    else:
        in_grad = output_gradient

    return in_grad.reshape(input_shape)


def swapaxes_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    *_, axis1, axis2 = inputs
    return output_gradient.swapaxes(axis1, axis2)


def variance_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    correction: float = 0.0,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1, 2])
    (input,) = inputs
    shape = input.shape
    if axis is None:
        n = np.prod(shape)
        return (
            (2 / (n - correction))
            * (input - np.mean(input, axis=axis, keepdims=True))
            * output_gradient
        )
    else:
        n = np.prod(np.array(shape)[np.array(axis)])
        if not keepdim:
            output_gradient = np.expand_dims(output_gradient, axis=axis)
        return (
            (2 / (n - correction))
            * (input - np.mean(input, axis=axis, keepdims=True))
            * output_gradient
        )


# def shape_grad(output_gradient: np.ndarray,
#                cache: CacheType,
#                idx: int,
#                *inputs: np.ndarray, -> np.ndarray:
#     if idx == 0:
#         return np.zeros_like(inputs[0])
#     else:
#         return gradient_exception(idx)


def reshape_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx, non_differentiables=[1])
    input, _ = inputs
    return output_gradient.reshape(input.shape)


def where_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    cond, *_ = inputs

    if idx == 1:
        return output_gradient * cond
    elif idx == 2:
        return output_gradient * ~cond

    raise RuntimeError("Numpy Where primitive grad operation. Something went wrong!")


def primitive_embedding_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    # TODO: Check this function, if it works properly add tests.
    verify_shapes(inputs, idx, non_differentiables=[0])
    if idx == 1:
        w_grad = np.zeros_like(inputs[1])
        np.add.at(w_grad, inputs[0], output_gradient)
        return w_grad
    else:
        raise RuntimeError("Something went wrong!")


def scaled_dot_product_attention_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | int | None = None,
):
    verify_shapes(inputs, idx, non_differentiables=[3, 4, 5, 6])
    if idx == 2:
        return matrix_multiplication_grad(
            output_gradient, None, 1, cache["attn_weight_soft_out"], inputs[2]
        )
    elif idx == 1:
        query, key, value, *_ = inputs
        attn_grad_1 = matrix_multiplication_grad(
            output_gradient, None, 0, cache["attn_weight_soft_out"], value
        )
        attn_grad_soft = softmax_grad(
            attn_grad_1,
            {"output": cache["attn_weight_soft_out"], "axis": -1},
            0,
            cache["attn_weight_soft_out"],
        )
        key_grad = matrix_multiplication_grad(
            attn_grad_soft, None, 1, query, np.swapaxes(key, -2, -1)
        )
        key_grad = np.swapaxes(key_grad, -2, -1) * cache["scale_factor"]
        return key_grad
    elif idx == 0:
        query, key, value, *_ = inputs
        attn_grad_1 = matrix_multiplication_grad(
            output_gradient, None, 0, cache["attn_weight_soft_out"], value
        )
        attn_grad_soft = softmax_grad(
            attn_grad_1,
            {"output": cache["attn_weight_soft_out"], "axis": -1},
            0,
            cache["attn_weight_soft_out"],
        )
        query_grad = matrix_multiplication_grad(
            attn_grad_soft,
            None,
            0,
            query,
            (np.swapaxes(key, -2, -1) * cache["scale_factor"]),
        )
        return query_grad
    else:
        raise RuntimeError("Something went wrong!")


def isnan_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    return output_gradient


def nan_to_num_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)
    if idx == 0:
        return ~(np.isnan(inputs[0]) | np.isinf(inputs[0])) * output_gradient
    raise RuntimeError(
        "Numpy nan_to_num primitive grad operation. Something went wrong!"
    )


# def index_grad(output_gradient: np.ndarray,
#                cache: CacheType,
#                idx: int,
#                *inputs: np.ndarray, -> np.ndarray:
#     if idx == 0:
#         input, index = inputs
#         grad = np.zeros(input.shape)
#         grad[index] = output_gradient
#         return grad
#     else:
#         return gradient_exception(idx)

# def sequence_slice_grad(output_gradient: np.ndarray,
#                      cache: CacheType,
#                      idx: int,
#                      *inputs: np.ndarray, -> np.ndarray:
#     if idx == 0:
#         input, *_ = inputs
#         start_idx = cache["start_idx"]
#         stop_idx = cache["stop_idx"]
#         step_size = cache["step_size"]
#         grad = np.zeros_like(input)
#         grad[start_idx: stop_idx: step_size] = output_gradient
#         return gradf
#     else:
#         return gradient_exception(idx)


def split_grad(
    output_gradient: list[np.ndarray],
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
):
    input, split_size, axis = inputs
    input_shape = input.shape
    grad_input = np.zeros(input_shape, dtype=output_gradient[0].dtype)

    # Calculate the split indices if `split_size` is an integer
    if isinstance(split_size, int):
        split_indices = [input_shape[axis] // split_size] * split_size
        if input_shape[axis] % split_size != 0:
            split_indices.append(input_shape[axis] % split_size)
    else:
        # Otherwise, use the list directly
        split_indices = split_size

    current_index = 0
    if axis < 0:
        axis = len(input_shape) + axis

    for grad, size in zip(output_gradient, split_indices, strict=False):
        slices = tuple(
            slice(current_index, current_index + size) if ax == axis else slice(None)
            for ax in range(len(input_shape))
        )
        grad_input[slices] = grad
        current_index += size

    return grad_input


def pad_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    input, padding = inputs

    slices = tuple(
        slice(p[0], p[0] + s) for p, s in zip(padding, input.shape, strict=False)
    )

    return output_gradient[slices]


def minus_grad(
    output_gradient: np.ndarray,
    cache: CacheType,
    idx: int,
    *inputs: np.ndarray,
) -> np.ndarray:
    verify_shapes(inputs, idx)

    return -output_gradient


primitive_grad_func_dict = {key: fn for key, fn in globals().items() if callable(fn)}
