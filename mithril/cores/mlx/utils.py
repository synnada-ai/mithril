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


from collections.abc import Callable, Sequence
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...common import BiMap
from ..utils import binary_search

dtype_map: BiMap[str, mx.Dtype] = BiMap(
    {
        "int8": mx.int8,
        "int16": mx.int16,
        "int32": mx.int32,
        "int64": mx.int64,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
        "bool": mx.bool_,  # type: ignore
    }
)


def find_optimal_sigmas(
    negative_dist_sq: mx.array, target_perplexity: mx.array, threshold: mx.array
) -> mx.array:
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role.
    Parameters
    ----------
    negative_dist_sq : mx.array
        Square of distance matrix multiplied by (-1).
    target_perplexity : float
        Desired perplexity value.
    Returns
    -------
    mx.array
        Returns optimal sigma values.
    """
    sigmas: list[float] = []

    # Make fn that returns perplexity of this row given sigma
    def eval_fn(sigma: float, i: int) -> mx.array:
        return perplexity_fn(negative_dist_sq[i, :], mx.array(sigma), i, threshold)

    # For each row of the matrix (each point in our dataset)
    for i in range(negative_dist_sq.shape[0]):
        eval_fn_p = partial(eval_fn, i=i)
        # Binary search over sigmas to achieve target perplexity
        low, high = binary_search(eval_fn_p, target_perplexity, lower=0.0)

        correct_sigma = (low + high) / 2
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return mx.array(sigmas, dtype=negative_dist_sq.dtype)


def log_sigmoid(
    input: mx.array, log: Callable[..., mx.array], robust: bool
) -> mx.array:
    min = mx.minimum(0, input)
    input = mx.exp(-mx.abs(input))
    if not robust:
        return min - mx.log1p(input)
    return min - log(1 + input)


def log_softmax(
    input: mx.array, log: Callable[..., mx.array], robust: bool, axis: int = -1
) -> mx.array:
    if not robust:
        return nn.log_softmax(input, axis)
    return input - log(mx.exp(input).sum(axis=axis, keepdims=True))


def calculate_binary_class_weight(labels: mx.array) -> mx.array:
    return (1 - labels.mean()) / labels.mean()


def calculate_categorical_class_weight(labels: mx.array, num_classes: int) -> mx.array:
    one_hot = mx.eye(num_classes)[labels]
    return calculate_class_weight(one_hot)


def calculate_class_weight(labels: mx.array) -> mx.array:
    return (
        (1 / labels.sum(axis=tuple(i for i in range(labels.ndim) if i != 1)))
        * labels.sum()
        / labels.shape[1]
    )


def calculate_cross_entropy_class_weights(
    input: mx.array,
    labels: mx.array,
    is_categorical: bool,
    weights: bool | list[float],
) -> mx.array:
    _weights = None
    if isinstance(weights, bool):
        if is_categorical:
            _weights = (
                calculate_categorical_class_weight(labels, input.shape[1]).astype(
                    input.dtype
                )
                if weights
                else mx.ones(input.shape[1], dtype=input.dtype)
            )
        else:
            _weights = (
                calculate_class_weight(labels).astype(input.dtype)
                if weights
                else mx.ones(input.shape[1], dtype=input.dtype)
            )
    else:
        _weights = mx.array(weights, dtype=input.dtype)
        if _weights.ndim > 1:
            raise ValueError(f"Provided weights: '{weights}' must be 1D list.")
    if not is_categorical:
        shape = [1 for _ in range(input.ndim)]
        shape[1] = input.shape[1]
        _weights = _weights.reshape(shape)
    return _weights


def get_submatrices1d(
    input: mx.array,
    output_size: Sequence[int],
    kernel_width_size: int,
    padding: int | tuple[int, int] = 0,
    stride: int = 1,
) -> mx.array:
    if isinstance(padding, tuple):
        input = mx.pad(input, ((0, 0), (0, 0), (padding[0], padding[1])))

    strides = [1]
    for idx, shape in enumerate(reversed(input.shape)):
        strides.append(strides[idx] * shape)
    strides = list(reversed(strides))[1:]

    out_w = output_size[-1]
    out_b, out_c, *_ = input.shape
    batch_str, channel_str, kern_w_str = strides

    return mx.as_strided(
        input,
        (out_b, out_c, out_w, kernel_width_size),
        (batch_str, channel_str, stride * kern_w_str, kern_w_str),
    )


def get_submatrices2d(
    input: mx.array,
    output_size: Sequence[int],
    kernel_height_size: int,
    kernel_width_size: int,
    padding: int | tuple[tuple[int, int], tuple[int, int]] = 0,
    stride: int = 1,
) -> mx.array:
    if isinstance(padding, tuple):
        input = mx.pad(
            input,
            (
                (0, 0),
                (0, 0),
                (padding[0][0], padding[0][1]),
                (padding[1][0], padding[1][1]),
            ),
        )

    strides = [1]
    for idx, shape in enumerate(reversed(input.shape)):
        strides.append(strides[idx] * shape)
    strides = list(reversed(strides))[1:]

    *_, out_h, out_w = output_size
    out_b, out_c, *_ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = strides

    return mx.as_strided(
        input,
        (out_b, out_c, out_h, out_w, kernel_height_size, kernel_width_size),
        (
            batch_str,
            channel_str,
            stride * kern_h_str,
            stride * kern_w_str,
            kern_h_str,
            kern_w_str,
        ),
    )


def perplexity_fn(
    negative_dist_sq: mx.array,
    sigmas: mx.array,
    zero_index: int | None,
    threshold: mx.array,
) -> mx.array:
    """Wrapper function for quick calculation of
        perplexity over a distance matrix.
    Parameters
    ----------
    negative_dist_sq : mx.array
        Square of distance matrix multiplied by (-1).
    sigmas : mx.array, optional
        Sigma values according to desired perplexity
        Sigmas calculated using binary search, by default None.
    zero_index : int, optional
        The index to be set 0, by default None.
    Returns
    -------
    float
        Returns current perplexity result.
    """
    prob_matrix = calc_prob_matrix(negative_dist_sq, sigmas, zero_index)
    prob_matrix = mx.clip(prob_matrix, threshold, (1 - threshold))
    entropy = -mx.sum(prob_matrix * mx.log2(prob_matrix), 1)
    perplexity = mx.power(2, entropy)
    if mx.isnan(perplexity).any():
        raise Exception(
            "Can not evaluate function (got a NaN value) somewhere"
            " n the given interval!"
        )
    return perplexity


def calc_prob_matrix(
    negative_dist_sq: mx.array, sigmas: mx.array, zero_index: int | None = None
) -> mx.array:
    """Convert a distances matrix to a matrix of probabilities.
    Parameters
    ----------
    negative_dist_sq : mx.array
        Square of distance matrix multiplied by (-1).
    sigmas : mx.array
        Sigma values according to desired perplexity
        Sigmas calculated using binary search.
    zero_index : int, optional
        The index to be set 0, by default None.
    Returns
    -------
    mx.array
        Returns conditional probabilities using distance matrix.
    """
    two_sig_sq = 2.0 * mx.square(sigmas.reshape((-1, 1)))

    broadcasted_shape = np.broadcast_shapes(negative_dist_sq.shape, two_sig_sq.shape)
    negative_dist_sq = mx.broadcast_to(negative_dist_sq, broadcasted_shape)
    two_sig_sq = mx.broadcast_to(two_sig_sq, broadcasted_shape)
    dist_sig = mx.where(
        two_sig_sq == 0,
        mx.zeros_like(mx.atleast_2d(negative_dist_sq)),
        mx.atleast_2d(negative_dist_sq) / two_sig_sq,
    )
    return tsne_softmax(dist_sig, diag_zero=True, zero_index=zero_index)


def tsne_softmax(
    input_tensor: mx.array,
    diag_zero: bool = False,
    zero_index: int | None = None,
) -> mx.array:
    input_tensor = input_tensor - mx.max(input_tensor, axis=1, keepdims=True)
    e = mx.exp(input_tensor)
    if zero_index is None:
        if diag_zero:
            e *= mx.ones_like(e) - mx.eye(len(e))
    else:
        # Since index is scalar, we have to deal with 1D arrays
        # in order to change its "zero_index"ed element. If we don't
        # squeeze, .at changes all the elements in that row for example
        # for 2D arrays.
        modified_ones = mx.ones_like(mx.squeeze(e))
        modified_ones[zero_index] = 0.0
        e *= modified_ones
    s = mx.sum(e, axis=1, keepdims=True)
    return e / s


def polynomial_features_helper(arr1: mx.array, arr2: mx.array) -> mx.array:
    # TODO: Consider using this function also in robust power.
    broadcasted_shape = np.broadcast_shapes(arr1.shape, arr2.shape)
    arr1 = mx.broadcast_to(arr1, broadcasted_shape)
    arr2 = mx.broadcast_to(arr2, broadcasted_shape)
    cond = mx.stop_gradient(mx.equal(arr1, 0) & mx.equal(arr2, 0))
    return mx.where(cond, mx.array(1.0, dtype=arr1.dtype), arr1**arr2)
