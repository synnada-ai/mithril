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


from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any

import numpy as np

from ....common import BiMap, find_dominant_type
from ...utils import binary_search, is_int_tuple_tuple

CacheType = dict[str, Any]

dtype_map: BiMap[str, Any] = BiMap(
    {
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "bool": np.bool_,
    }
)


def get_submatrices1d(
    input: np.ndarray[Any, Any],
    output_size: tuple[int, ...],
    kernel_width_size: int,
    padding: tuple[int, int] = (0, 0),
    stride: int = 1,
    dilate: int = 0,
) -> np.ndarray[Any, Any]:
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)

    # pad the input if necessary
    if working_pad != (0, 0):
        working_input = np.pad(
            working_input,
            pad_width=((0, 0), (0, 0), (working_pad[0], working_pad[1])),
            mode="constant",
            constant_values=(0.0,),
        )

    *_, out_w = output_size
    out_b, out_c, *_ = input.shape
    batch_str, channel_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_w, kernel_width_size),
        (batch_str, channel_str, stride * kern_w_str, kern_w_str),
    )


# TODO: padding, strinde and dilation must be int or tuple.
def get_submatrices2d(
    input: np.ndarray[Any, Any],
    output_size: tuple[int, ...],
    kernel_height_size: int,
    kernel_width_size: int,
    padding: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0)),
    stride: int = 1,
    dilate: int = 0,
) -> np.ndarray[Any, Any]:
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if is_int_tuple_tuple(padding):
        working_input = np.pad(
            working_input,
            pad_width=(
                (0, 0),
                (0, 0),
                (working_pad[0][0], working_pad[0][1]),
                (working_pad[1][0], working_pad[1][1]),
            ),
            mode="constant",
            constant_values=(0.0,),
        )

    *_, out_h, out_w = output_size
    out_b, out_c, *_ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
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


def tsne_softmax(
    input_tensor: np.ndarray[Any, Any],
    diag_zero: bool = False,
    zero_index: int | None = None,
) -> np.ndarray[Any, Any]:
    input_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
    e = np.exp(input_tensor)
    if zero_index is None:
        if diag_zero:
            np.fill_diagonal(e, 0.0)
    else:
        e[:, zero_index] = 0.0
    s = np.sum(e, axis=1, keepdims=True)
    return e / s


def calc_prob_matrix(
    negative_dist_sq: np.ndarray[Any, Any],
    sigmas: np.ndarray[Any, Any],
    zero_index: int | None = None,
) -> np.ndarray[Any, Any]:
    """Convert a distances matrix to a matrix of probabilities.
    Parameters
    ----------
    negative_dist_sq : np.ndarray
        Square of distance matrix multiplied by (-1).
    sigmas : np.ndarray
        Sigma values according to desired perplexity
        Sigmas calculated using binary search.
    zero_index : int, optional
        The index to be set 0, by default None.
    Returns
    -------
    np.ndarray
        Returns conditional probabilities using distance matrix.
    """
    two_sig_sq = 2.0 * np.square(sigmas.reshape((-1, 1)))
    if two_sig_sq.shape[0] == 1:
        dist_sig = [negative_dist_sq / two_sig_sq, np.array(0.0)][
            np.squeeze(two_sig_sq) == 0.0
        ]
    else:
        mask = two_sig_sq == 0.0
        dist_sig = np.zeros_like(negative_dist_sq)
        dist_sig[~mask[:, 0], :] = negative_dist_sq[~mask[:, 0], :] / two_sig_sq[~mask]
    return tsne_softmax(dist_sig, diag_zero=True, zero_index=zero_index)


def perplexity_fn(
    negative_dist_sq: np.ndarray[Any, Any],
    sigmas: np.ndarray[Any, Any],
    zero_index: int,
    threshold: float,
) -> np.ndarray[Any, Any]:
    """Wrapper function for quick calculation of
        perplexity over a distance matrix.
    Parameters
    ----------
    negative_dist_sq : np.ndarray
        Square of distance matrix multiplied by (-1).
    sigmas : np.ndarray, optional
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
    prob_matrix = np.clip(prob_matrix, threshold, (1 - threshold))
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2**entropy
    return perplexity


def find_optimal_sigmas(
    negative_dist_sq: np.ndarray[Any, Any], target_perplexity: int, threshold: float
) -> np.ndarray[Any, Any]:
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role.
    Parameters
    ----------
    negative_dist_sq : np.ndarray
        Square of distance matrix multiplied by (-1).
    target_perplexity : float
        Desired perplexity value.
    Returns
    -------
    np.ndarray
        Returns optimal sigma values.
    """
    sigmas: list[float] = []

    # Make fn that returns perplexity of this row given sigma
    def eval_fn(sigma: float, i: int) -> np.ndarray[Any, Any]:
        return perplexity_fn(negative_dist_sq[i, :], np.array(sigma), i, threshold)

    # For each row of the matrix (each point in our dataset)
    for i in range(negative_dist_sq.shape[0]):
        eval_fn_p = partial(eval_fn, i=i)

        # Binary search over sigmas to achieve target perplexity
        low, high = binary_search(eval_fn_p, target_perplexity, lower=0.0)  # type: ignore
        correct_sigma = (low + high) / 2
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


def log_sigmoid(
    input: np.ndarray[Any, Any], log: Callable[..., np.ndarray[Any, Any]], robust: bool
) -> np.ndarray[Any, Any]:
    min = np.minimum(0, input)
    input = np.exp(-np.abs(input))
    if not robust:
        return min - np.log1p(input)
    return min - log(1 + input)


def log_softmax(
    input: np.ndarray[Any, Any],
    log: Callable[..., np.ndarray[Any, Any]],
    robust: bool,
    axis: int = -1,
) -> np.ndarray[Any, Any]:
    return input - log(np.exp(input).sum(axis=axis, keepdims=True))


def calculate_binary_class_weight(labels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return (1 - labels.mean()) / labels.mean()


def calculate_categorical_class_weight(
    labels: np.ndarray[Any, Any], num_classes: int
) -> np.ndarray[Any, Any]:
    one_hot = np.eye(num_classes)[labels]
    return calculate_class_weight(one_hot)


def calculate_class_weight(labels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return (
        (1 / labels.sum(axis=tuple(i for i in range(labels.ndim) if i != 1)))
        * labels.sum()
        / labels.shape[1]
    )


def calculate_cross_entropy_class_weights(
    input: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    is_categorical: bool,
    weights: bool | list[float],
) -> np.ndarray[Any, Any]:
    _weights: np.ndarray[Any, Any]
    if isinstance(weights, bool):
        if is_categorical:
            _weights = (
                calculate_categorical_class_weight(labels, input.shape[1]).astype(
                    input.dtype
                )
                if weights
                else np.ones(input.shape[1], dtype=input.dtype)
            )
        else:
            _weights = (
                calculate_class_weight(labels).astype(input.dtype)
                if weights
                else np.ones(input.shape[1], dtype=input.dtype)
            )
    else:
        _weights = np.array(weights, dtype=input.dtype)
        if _weights.ndim > 1:
            raise ValueError(f"Provided weights: '{weights}' must be 1D list.")
    if not is_categorical:
        shape = [1 for _ in range(input.ndim)]
        shape[1] = input.shape[1]
        _weights = _weights.reshape(shape)
    return _weights


# TODO: resolve its types
def write_into_cache[T: np.ndarray[Any, Any] | tuple[Any, ...] | int | float](
    cache: CacheType | None,
    key: str,
    value: T,
    *,
    constant: bool = False,
    func: Callable[..., Any] | None = None,
) -> T:
    """Writes key-value pair into the provided cache if there exists.
    If func given it is called with given value and then
    result is written into cache for given key. If constant
    flag is set, it is written (and func is called) only once
    into the cache for efficiency.

    Parameters
    ----------
    cache : _type_
        _description_
    key : _type_
        _description_
    value : _type_
        _description_
    constant : bool, optional
        _description_, by default False
    func : _type_, optional
        _description_, by default None
    args : _type_, optional
        _description_, by default None
    """
    if cache is None:
        # Cache is None in inference modes of the models.
        if func is None:
            return value
        return func(*value) if isinstance(value, tuple | list) else func(value)
    elif func is None:
        result = value
        cache[key] = result
    elif not (constant and key in cache):
        result = func(*value) if isinstance(value, tuple | list) else func(value)
        cache[key] = result
    else:
        result = cache[key]
    # TODO: Resolve here
    return result


def accumulate_grads(
    gradient: np.ndarray[Any, Any],
    input: np.ndarray[Any, Any],
    cache: CacheType | None,
    idx: int,
) -> np.ndarray[Any, Any]:
    axes = write_into_cache(
        cache,
        "accumulate" + str(idx),
        (gradient.shape, input.shape),
        constant=True,
        func=_accumulate_grads_helper,
    )
    # TODO: raise exception for len(gradient.shape) < len(input.shape) condition
    if axes and not len(gradient.shape) < len(input.shape):
        return np.reshape(np.sum(gradient, axis=axes), input.shape)
    else:
        return gradient


def _accumulate_grads_helper(
    grad_shape: tuple[int, ...], input_shape: tuple[int, ...]
) -> tuple[int, ...]:
    # TODO: Refactor the code below
    rev_grad = list(reversed(grad_shape))
    axes = tuple([i for i in range(len(grad_shape) - len(input_shape))])
    for idx, item in enumerate(reversed(input_shape)):
        if item > rev_grad[idx]:
            raise ValueError(
                "Input shape could not be larger than output gradient shape!"
            )
        if (item != rev_grad[idx]) and rev_grad[idx] != 1:
            axes += (len(grad_shape) - (idx + 1),)
    return axes


def calc_input_slices(
    output_gradient: np.ndarray[Any, Any],
    axis: int | None,
    args: Sequence[np.ndarray[Any, Any]],
) -> dict[int, tuple[slice, ...]]:
    # Calculates the slices of output_gradient corresponding to
    # inputs.
    slices: dict[int, tuple[slice, ...]] = {}
    base_slices = [slice(None)] * output_gradient.ndim
    finish = 0
    for idx, arg in enumerate(args):
        start = finish
        finish = start + (arg.shape[axis] if axis is not None else arg.size)
        current_slice = base_slices.copy()
        current_slice[axis if axis is not None else 0] = slice(start, finish, None)
        slices[idx] = tuple(current_slice)
    return slices


def verify_shapes(
    inputs: tuple[np.ndarray[Any, Any], ...],
    idx: int,
    non_differentiables: Iterable[int] | None = None,
) -> None:
    if idx >= len(inputs):
        raise Exception(f"Gradient is not defined for the input at index {idx}!")
    if non_differentiables is not None and idx in non_differentiables:
        raise Exception(f"Given key at index {idx} is not differentiable!")


def determine_dtype(
    input: Any, dtype: str | None, default_dtype: str, precision: int
) -> str:
    if isinstance(dtype, str):
        return dtype

    if isinstance(input, (np.ndarray | np.generic)):
        dtype_name = "".join(char for char in str(input.dtype) if not char.isdigit())
    else:
        dtype_name = find_dominant_type(input).__name__

    return dtype_name + str(precision) if dtype_name != "bool" else "bool"
