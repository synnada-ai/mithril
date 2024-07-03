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
from typing import Any, overload

import numpy as np

from .... import core
from ....utils.type_utils import is_int_tuple_tuple
from ....utils.utils import binary_search, find_dominant_type

ArrayType = np.ndarray

dtype_map = {
    "int16": np.int16,
    "int32": np.int32,
    "int": np.int32,
    "int64": np.int64,
    "long": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float": np.float32,
    "float64": np.float64,
    "double": np.float64,
    "bool": np.bool_,
    None: None,
}

CacheType = dict[str, Any]


@overload
def write_into_cache(
    cache: CacheType | None,
    key: str,
    value: tuple[np.ndarray, ...],
    *,
    constant: bool = False,
    func: Callable | None = None,
) -> tuple[np.ndarray, ...]: ...


@overload
def write_into_cache(
    cache: CacheType | None,
    key: str,
    value: tuple[Any, ...],
    *,
    constant: bool = False,
    func: Callable | None = None,
) -> tuple[Any, ...]: ...


@overload
def write_into_cache(
    cache: CacheType | None,
    key: str,
    value: int | float,
    *,
    constant: bool = False,
    func: Callable | None = None,
) -> int | float: ...


@overload
def write_into_cache(
    cache: CacheType | None,
    key: str,
    value: np.ndarray,
    *,
    constant: bool = False,
    func: Callable | None = None,
) -> np.ndarray: ...


# TODO: resolve its types
def write_into_cache(
    cache: CacheType | None,
    key: str,
    value: np.ndarray | tuple[np.ndarray, ...] | int | float,
    *,
    constant: bool = False,
    func: Callable | None = None,
) -> np.ndarray | tuple[np.ndarray, ...] | int | float:
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
        if func is None:
            result = value
        else:
            result = func(*value) if isinstance(value, tuple | list) else func(value)
        cache[key] = result
    else:
        result = cache[key]
    # TODO: Resolve here
    return result  # type: ignore


def get_submatrices1d(
    input: np.ndarray,
    output_size,
    kernel_width_size,
    padding: tuple[int, int] = (0, 0),
    stride=1,
    dilate=0,
):  # TODO: Return type???
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
    input: np.ndarray,
    output_size,
    kernel_height_size,
    kernel_width_size,
    padding: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0)),
    stride=1,
    dilate=0,
):  # TODO: Return type???
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
    input_tensor: np.ndarray,
    diag_zero: bool = False,
    zero_index: int | None = None,
) -> np.ndarray:
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
    negative_dist_sq: np.ndarray, sigmas: np.ndarray, zero_index=None
) -> np.ndarray:
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
        dist_sig = [negative_dist_sq / two_sig_sq, 0][np.squeeze(two_sig_sq) == 0.0]
    else:
        mask = two_sig_sq == 0.0
        dist_sig = np.zeros_like(negative_dist_sq)
        dist_sig[~mask[:, 0], :] = negative_dist_sq[~mask[:, 0], :] / two_sig_sq[~mask]
    return tsne_softmax(dist_sig, diag_zero=True, zero_index=zero_index)


def perplexity_fn(
    negative_dist_sq: np.ndarray,
    sigmas: np.ndarray,
    zero_index: int,
    threshold: float,
) -> np.ndarray:
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
    negative_dist_sq: np.ndarray, target_perplexity: int, threshold: float
) -> np.ndarray:
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
    sigmas = []

    # Make fn that returns perplexity of this row given sigma
    def eval_fn(sigma, i):
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


# Shared or reused intermediate value calculator functions for primitive models


def find_label_indices(input_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(len(input_array)), input_array.T)


def calc_input_slices(
    output_gradient: np.ndarray, axis: int, *args: np.ndarray
) -> dict[str, tuple[slice, ...]]:
    # Calculates the slices of output_gradient corresponding to
    # inputs.
    slices = {}
    base_slices = [slice(None)] * output_gradient.ndim
    finish = 0
    for idx, arg in enumerate(args):
        start = finish
        finish = start + (arg.shape[axis] if axis is not None else arg.size)
        current_slice = base_slices.copy()
        current_slice[axis if axis is not None else 0] = slice(start, finish, None)
        slices[f"input{idx + 1}"] = tuple(current_slice)
    return slices


def handle_dtype(dtype: Any) -> Any:
    if isinstance(dtype, core.Dtype):
        return dtype_map[dtype.name]
    elif isinstance(dtype, str) and dtype in dtype_map:
        return dtype_map[dtype]
    else:
        try:
            return np.dtype(dtype)
        except TypeError as err:
            raise TypeError(f"Provided data type '{dtype}' not understood") from err


def creation_fn_wrapper(*args, fn: Callable, precision: int, dtype=None, **kwargs):
    if dtype is not None:
        dtype = handle_dtype(dtype)
        data = fn(*args, dtype=dtype, **kwargs)
    else:
        data = fn(*args, **kwargs)
        data = handle_data_precision(data, precision=precision)
    return data


def conversion_fn_wrapper(
    data, *args, fn: Callable, precision: int, dtype=None, **kwargs
):
    if dtype is not None:
        dtype = handle_dtype(dtype)
    if isinstance(data, ArrayType):
        if dtype is not None:
            return data.astype(dtype)
        return handle_data_precision(data, precision=precision)
    else:
        _data = fn(data, *args, dtype=dtype, **kwargs)
        if dtype is None:
            return handle_data_precision(_data, precision=precision)
        return _data


def handle_data_precision(data: ArrayType, precision: int) -> ArrayType:
    if isinstance(data, float | int):
        return data
    _dtype = data.dtype
    # Do not make any changes to boolean types.
    if _dtype != np.bool_:
        if np.issubdtype(_dtype, np.integer) and _dtype != getattr(
            np, f"int{precision}"
        ):
            data = data.astype(f"int{precision}")
        elif np.issubdtype(_dtype, np.floating) and _dtype != getattr(
            np, f"float{precision}"
        ):
            data = data.astype(f"float{precision}")
    return data


def handle_data_dtype(data: np.ndarray, dtype: core.Dtype | int) -> np.ndarray:
    if isinstance(dtype, int):
        dtype = core.Dtype(dtype)

    if data.dtype != dtype_map[dtype.name]:
        return data.astype(dtype_map[dtype.name])
    return data


def make_array(input: int | float | ArrayType, precision):
    return handle_data_precision(np.array(input), precision=precision)


def accumulate_grads(gradient: np.ndarray, input: np.ndarray, cache, idx):
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


def _accumulate_grads_helper(grad_shape, input_shape):
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


def log_sigmoid(input: np.ndarray, log: Callable, robust: bool):
    min = np.minimum(0, input)
    input = np.exp(-np.abs(input))
    if not robust:
        return min - np.log1p(input)
    return min - log(1 + input)


def log_softmax(input: np.ndarray, log: Callable, robust: bool, axis: int = -1):
    return input - log(np.exp(input).sum(axis=axis, keepdims=True))


def calculate_binary_class_weight(labels):
    return (1 - labels.mean()) / labels.mean()


def calculate_categorical_class_weight(labels, num_classes: int):
    one_hot = np.eye(num_classes)[labels]
    return calculate_class_weight(one_hot)


def calculate_class_weight(labels):
    return (
        (1 / labels.sum(axis=tuple(i for i in range(labels.ndim) if i != 1)))
        * labels.sum()
        / labels.shape[1]
    )


def calculate_cross_entropy_class_weights(
    input: np.ndarray,
    labels: np.ndarray,
    is_categorical: bool,
    weights: bool | list[float],
):
    _weights = None
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


def get_type(input: int | float | bool | Sequence, precision: int):
    type = find_dominant_type(input).__name__
    if type == "bool":
        return np.bool_

    return getattr(np, type + str(precision))


def verify_shapes(
    inputs: tuple[np.ndarray, ...], idx: int, non_differentiables=None
) -> None:
    if idx >= len(inputs):
        raise Exception(f"Gradient is not defined for the input at index {idx}!")
    if non_differentiables is not None and idx in non_differentiables:
        raise Exception(f"Given key at index {idx} is not differentiable!")
