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
from typing import Any

import jax
import jax.numpy as jnp
from jax import vmap

from .... import core
from ....utils.utils import binary_search, find_dominant_type

ArrayType = jax.Array

dtype_map = {
    "int16": jnp.int16,
    "int32": jnp.int32,
    "int": jnp.int32,
    "int64": jnp.int64,
    "long": jnp.int64,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "float": jnp.float32,
    "float64": jnp.float64,
    "double": jnp.float64,
    "bool": jnp.bool_,
    None: None,
}


def broadcast_to_highest(
    input1: jax.Array, input2: jax.Array
) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    out_shape = jnp.broadcast_shapes(input1.shape, input2.shape)
    return (
        jnp.atleast_1d(jnp.broadcast_to(input1, out_shape)),
        jnp.atleast_1d(jnp.broadcast_to(input2, out_shape)),
        out_shape,
    )


def vmapper(func: Callable, count: int) -> Callable:
    for _ in range(count):
        func = jax.vmap(func)
    return func


# Unlike the backends of numpy and torch, grad value at 0 is not zero in jnp.abs().
# This was causing problems when input is exactly zero under threshold. Therefore,
# jnp.abs() is rewritten as jnp.maximum(input1,-input1) in order to handle such cases.
def robust_power_under_threshold(
    input1: jax.Array, input2: jax.Array, threshold: jax.Array
) -> jax.Array:
    # slope = input2 * (threshold ** (input2))
    # return threshold ** input2 + slope * (input1/threshold - 1)
    return (1 / threshold) * jnp.maximum(input1, -input1)


def robust_power_above_threshold(
    input1: jax.Array, input2: jax.Array, threshold: jax.Array
) -> jax.Array:
    return jnp.abs(input1) ** input2


def robust_power_helper(
    input1: jax.Array, input2: jax.Array, threshold: jax.Array
) -> jax.Array:
    def cond_fun(cond, input1, input2):
        return jax.lax.cond(
            cond,
            robust_power_under_threshold,
            robust_power_above_threshold,
            input1,
            input2,
            threshold,
        )

    condition = (
        input2
        < (
            1
            - jnp.log(threshold)
            / (jnp.log(threshold) + jnp.log(jnp.abs(input1) / threshold))
        )
    ) & (jnp.abs(input1) < threshold)
    return vmap(cond_fun)(condition, input1, input2)


def robust_log_helper(input1: jax.Array, threshold: jax.Array) -> jax.Array:
    def cond_fun(cond, input1):
        return jax.lax.cond(
            cond,
            lambda x: jnp.log(threshold) + (jnp.abs(x) / threshold) - 1.0,
            lambda x: jnp.log(jnp.abs(x)),
            input1,
        )

    condition = jnp.abs(input1) < threshold
    return jax.vmap(cond_fun)(condition, input1)


def stable_reciprocal_helper(input1: jax.Array, threshold: jax.Array) -> jax.Array:
    def cond_fun(cond, input1):
        return jax.lax.cond(
            cond,
            lambda x: -x / jnp.square(threshold)
            + (2 / threshold) * (jnp.sign(x) + (1 - jnp.sign(jnp.abs(x)))),
            lambda x: jnp.reciprocal(x),
            input1,
        )

    condition = jnp.abs(input1) < threshold
    return jax.vmap(cond_fun)(condition, input1)


def robust_sqrt_helper(input1: jax.Array, threshold: jax.Array) -> jax.Array:
    def cond_fun(cond, input1):
        return jax.lax.cond(
            cond,
            lambda x: jnp.abs(x) * jnp.reciprocal(jnp.sqrt(threshold)),
            lambda x: jnp.sqrt(jnp.abs(x)),
            input1,
        )

    condition = jnp.abs(input1) < threshold
    return jax.vmap(cond_fun)(condition, input1)


def many_to_one_inference_helper(
    prev_hidden: jax.Array,
    input: jax.Array,
    w_ih: jax.Array,
    w_hh: jax.Array,
    w_ho: jax.Array,
    bias_h: jax.Array,
    bias_o: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    term_1 = prev_hidden @ w_hh
    term_2 = input @ w_ih
    term_3 = term_1 + term_2 + bias_h
    hidden = jnp.tanh(term_3)
    term_4 = hidden @ w_ho
    out = term_4 + bias_o
    return hidden, out


def tsne_softmax(
    input_tensor: jax.Array,
    diag_zero: bool = False,
    zero_index: int | None = None,
) -> jax.Array:
    input_tensor = input_tensor - jnp.max(input_tensor, axis=1, keepdims=True)
    e = jnp.exp(input_tensor)
    if zero_index is None:
        if diag_zero:
            e *= jnp.ones_like(e, dtype=input_tensor.dtype) - jnp.eye(
                len(e), dtype=input_tensor.dtype
            )
    else:
        # Since index is scalar, we have to deal with 1D arrays
        # in order to change its "zero_index"ed element. If we don't
        # squeeze, .at changes all the elements in that row for example
        # for 2D arrays.
        modified_ones = jnp.ones_like(jnp.squeeze(e), dtype=input_tensor.dtype)
        modified_ones = modified_ones.at[zero_index].set(0.0)
        e *= modified_ones
    s = jnp.sum(e, axis=1, keepdims=True)
    return e / s


def calc_prob_matrix(
    negative_dist_sq: jax.Array, sigmas: jax.Array, zero_index=None
) -> jax.Array:
    """Convert a distances matrix to a matrix of probabilities.
    Parameters
    ----------
    negative_dist_sq : jax.Array
        Square of distance matrix multiplied by (-1).
    sigmas : jax.Array
        Sigma values according to desired perplexity
        Sigmas calculated using binary search.
    zero_index : int, optional
        The index to be set 0, by default None.
    Returns
    -------
    jax.Array
        Returns conditional probabilities using distance matrix.
    """
    two_sig_sq = 2.0 * jnp.square(sigmas.reshape((-1, 1)))
    dist_sig = vmap(jax.lax.cond, in_axes=(0, None, None, 0, 0))(
        jnp.squeeze(two_sig_sq == 0, axis=1),
        lambda a, b: jnp.zeros_like(a, dtype=negative_dist_sq.dtype),
        lambda a, b: a / b,
        jnp.atleast_2d(negative_dist_sq),
        two_sig_sq,
    )
    return tsne_softmax(dist_sig, diag_zero=True, zero_index=zero_index)


def perplexity_fn(
    negative_dist_sq: jax.Array,
    sigmas: jax.Array,
    zero_index: int | None,
    threshold: jax.Array,
) -> jax.Array:
    """Wrapper function for quick calculation of
        perplexity over a distance matrix.
    Parameters
    ----------
    negative_dist_sq : jax.Array
        Square of distance matrix multiplied by (-1).
    sigmas : jax.Array, optional
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
    prob_matrix = jnp.clip(prob_matrix, threshold, (1 - threshold))
    entropy = -jnp.sum(prob_matrix * jnp.log2(prob_matrix), 1)
    perplexity = 2**entropy
    return perplexity


def find_optimal_sigmas(
    negative_dist_sq: jax.Array, target_perplexity: jax.Array, threshold: jax.Array
) -> jax.Array:
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role.
    Parameters
    ----------
    negative_dist_sq : jax.Array
        Square of distance matrix multiplied by (-1).
    target_perplexity : float
        Desired perplexity value.
    Returns
    -------
    jax.Array
        Returns optimal sigma values.
    """
    sigmas = []

    # For each row of the matrix (each point in our dataset)
    for i in range(negative_dist_sq.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        def eval_fn(sigma):
            return perplexity_fn(negative_dist_sq[i, :], jnp.array(sigma), i, threshold)  # noqa: B023

        # Binary search over sigmas to achieve target perplexity
        low, high = binary_search(eval_fn, target_perplexity, lower=0.0)
        correct_sigma = (low + high) / 2
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return jnp.array(sigmas, dtype=negative_dist_sq.dtype)


def polynomial_features_helper(x, y):
    # NOTE: This helper function is used to handle (0.0 ** 0) case. JAX original
    # power function gradient returns NAN for this case but we want the gradient
    # return 0.0 without changing forward characteristics for this point.
    # TODO: Consider using this function also in robust power.
    return jax.lax.cond(
        (x == 0.0) & (y == 0),
        lambda x, y: jnp.array(1.0, dtype=x.dtype),
        lambda x, y: x**y,
        x,
        y,
    )


def get_available_devices():
    backends = set(jax._src.xla_bridge.backends()) - set(["interpreter"])  # type: ignore
    devices = [
        f"{backend.replace('METAL','mps')}:{idx}"
        for backend in list(backends)
        for idx in range(jax.device_count(backend))
    ]
    return devices


def get_device(device: str):
    backend, device_idx = _parse_device_string(device)
    filtered_list = list(
        filter(
            lambda x: x.casefold() == backend.casefold(),
            _get_available_backends(),
        )
    )

    if len(filtered_list) == 0 or len(devices := jax.devices(backend)) < device_idx:
        raise RuntimeError(
            f"Specified device: '{device}' is not available!"
            f"Available devices: {get_available_devices()}"
        )

    return devices[device_idx]


def _get_available_backends() -> list[str]:
    backends = set(jax._src.xla_bridge.backends()) - set(["interpreter"])  # type: ignore
    return list(backends)


def _parse_device_string(device: str):
    device_parts = device.split(":")
    backend = device_parts[0].replace("mps", "METAL")
    device_idx = 0
    if len(device_parts) > 1:
        try:
            device_idx = int(device_parts[1])
        except ValueError as err:
            raise ValueError(
                f"Specified device: '{device}' is not available!"
                f"Available devices: {get_available_devices()}"
            ) from err

    return backend, device_idx


def handle_dtype(dtype: Any) -> Any:
    if isinstance(dtype, core.Dtype):
        return dtype_map[dtype.name]
    elif isinstance(dtype, str) and dtype in dtype_map:
        return dtype_map[dtype]
    else:
        try:
            return jnp.dtype(dtype)
        except TypeError as err:
            raise TypeError(f"Provided data type '{dtype}' not understood") from err


def creation_fn_wrapper(
    *args, fn: Callable, dtype=None, device: str, precision: int, **kwargs
):
    if isinstance(device, str):
        _device = get_device(device)

    if dtype is not None:
        dtype = handle_dtype(dtype)
        with jax.default_device(_device):
            data = fn(*args, dtype=dtype, **kwargs)
    else:
        with jax.default_device(_device):
            data = fn(*args, **kwargs)
            data = handle_data_precision(data, precision)
    return data


def conversion_fn_wrapper(
    data, *args, fn: Callable, device: str, precision: int, dtype=None, **kwargs
):
    if isinstance(device, str):
        _device = get_device(device)

    if dtype is not None:
        dtype = handle_dtype(dtype)
    if isinstance(data, ArrayType):
        if next(iter(data.devices())) != _device:
            data = jax.device_put(data, _device)
        if dtype is not None:
            return data.astype(dtype)
        return handle_data_precision(data, precision)
    else:
        with jax.default_device(_device):
            _data = fn(data, *args, dtype=dtype, **kwargs)
        if dtype is None:  # User did not specify dtype explicitly
            return handle_data_precision(_data, precision)
        return _data


def handle_data_precision(data: ArrayType, precision: int) -> ArrayType:
    _dtype = data.dtype
    # Do not make any changes to boolean types.
    if _dtype != jax.numpy.bool_:
        if jax.numpy.issubdtype(_dtype, jax.numpy.integer) and _dtype != getattr(
            jax.numpy, f"int{precision}"
        ):
            data = data.astype(f"int{precision}")
        elif jax.numpy.issubdtype(_dtype, jax.numpy.floating) and _dtype != getattr(
            jax.numpy, f"float{precision}"
        ):
            data = data.astype(f"float{precision}")
    return data


def handle_data_dtype(data: jax.Array, dtype: core.Dtype | int) -> jax.Array:
    if isinstance(dtype, int):
        dtype = core.Dtype(dtype)

    if data.dtype != dtype_map[dtype.name]:
        return data.astype(dtype_map[dtype.name])
    return data


def get_type(input: int | float | bool | Sequence, precision: int):
    type = find_dominant_type(input).__name__
    if type == "bool":
        return jax.numpy.bool_

    return getattr(jax.numpy, type + str(precision))


def calculate_tpr_fpr(threshold, input, label):
    input_c = input.copy()

    n_positive = (label == 1).sum()
    n_negative = len(label) - n_positive

    input_c = jax.numpy.where(input_c >= threshold, 1, 0)
    true_positives = jnp.sum((input_c == 1) & (label == 1))
    false_positives = jnp.sum((input_c == 1) & (label == 0))

    fpr = false_positives / n_negative
    tpr = true_positives / n_positive
    return tpr, fpr


def log_sigmoid(input: jax.Array, log: Callable, robust: bool):
    min = jnp.minimum(0, input)
    input = jnp.exp(-jnp.abs(input))
    if not robust:
        return min - jnp.log1p(input)
    return min - log(1 + input)


def log_softmax(input: jax.Array, log: Callable, robust: bool, axis: int = -1):
    if not robust:
        return jax.nn.log_softmax(input, axis)
    return input - log(jnp.exp(input).sum(axis=axis, keepdims=True))


def calculate_binary_class_weight(labels) -> jax.Array:
    return (1 - labels.mean()) / labels.mean()


def calculate_categorical_class_weight(labels, num_classes: int):
    one_hot = jnp.eye(num_classes)[labels]
    return calculate_class_weight(one_hot)


def calculate_class_weight(labels):
    # Expected shape (N, C, ...) or (C)
    return (
        (1 / labels.sum(axis=tuple(i for i in range(labels.ndim) if i != 1)))
        * labels.sum()
        / labels.shape[1]
    )


def calculate_cross_entropy_class_weights(
    input: jax.Array,
    labels: jax.Array,
    is_categorical: bool,
    weights: bool | list[float],
):
    _weights = None
    with jax.default_device(next(iter(labels.devices()))):
        if isinstance(weights, bool):
            if is_categorical:
                _weights = (
                    calculate_categorical_class_weight(labels, input.shape[1]).astype(
                        input.dtype
                    )
                    if weights
                    else jnp.ones(input.shape[1], dtype=input.dtype)
                )
            else:
                _weights = (
                    calculate_class_weight(labels).astype(input.dtype)
                    if weights
                    else jnp.ones(input.shape[1], dtype=input.dtype)
                )
        else:
            _weights = jnp.array(weights, dtype=input.dtype)
            if _weights.ndim > 1:
                raise ValueError(f"Provided weights: '{weights}' must be 1D list.")
    if not is_categorical:
        shape = [1 for _ in range(input.ndim)]
        shape[1] = input.shape[1]
        _weights = _weights.reshape(shape)
    return _weights
