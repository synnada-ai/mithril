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

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from ....core import Dtype
from ...backend import Backend, PadWidthType
from ...utils import process_shape
from . import ops, ops_grad, utils


class NumpyBackend(Backend[np.ndarray]):
    """A backend implementation for the Mithril library using NumPy with
    manual gradient support.

    Parameters
    ----------
    device: str, optional
        The device on which to perform computations, default is "cpu".
    precision: int, optional
        The precision of the arrays, either 32 or 64, default is 32.
    """

    type = "numpy"

    registered_primitives = {}
    primitive_fn_path = "mithril.backends.with_manualgrad.numpy_backend.ops"
    primitive_grad_fn_path = "mithril.backends.with_manualgrad.numpy_backend.ops_grad"
    registered_primitives_grad_fn: dict[str, Callable] = {}

    def __init__(self, device: str = "cpu", precision: int = 32) -> None:
        self._precision = precision
        if device != "cpu":
            raise RuntimeError(
                f"Specified device: '{device}' is not available!"
                f"Available devices: {NumpyBackend.get_available_devices()}"
            )
        self._device = device

        super().__init__()

        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict
        self.primitive_grad_function_dict = ops_grad.primitive_grad_func_dict
        np.random.seed(self.seed)

    @property
    def is_manualgrad(self):
        return True

    @property
    def inf(self):
        return np.inf

    @property
    def DataType(self):  # noqa: N802
        return utils.ArrayType

    def get_backend_array_type(self):
        return np.ndarray

    @staticmethod
    def get_available_devices() -> list[str]:
        """Static method to get available devices. Currently, in the NumpyBackend,
        only the "cpu" device is supported.

        Parameters
        ----------
        list[str]
            List of available devices.
        """
        return ["cpu"]

    @staticmethod
    def register_primitive(fn: Callable, fn_grad: Callable) -> None:  # type: ignore[override]
        formula_key = fn.__name__
        NumpyBackend.registered_primitives[formula_key] = fn
        NumpyBackend.registered_primitives_grad_fn[formula_key + "_grad"] = fn_grad

    def set_seed(self, seed: int):
        self.seed = seed
        np.random.seed(seed)

    def _creation_fn_wrapper(self, fn: Callable) -> Callable:
        """
        Wrapper for NumPy array creation functions.

        Parameters
        ----------
        fn: Callable
            The original array creation function.

        Returns
        -------
        Callable
            A wrapped function that creates NumPy arrays with specified dtype.

        Notes
        -----
        This wrapper ensures that NumPy arrays are created with the correct dtype.
        """
        return partial(utils.creation_fn_wrapper, fn=fn, precision=self.precision)

    def _conversion_fn_wrapper(self, fn: Callable) -> Callable:
        """
        Wrapper for NumPy array conversion functions.

        Parameters
        ----------
        fn: Callable
            The original array conversion function.

        Returns
        -------
        Callable
            A wrapped function that converts arrays to NumPy arrays with
            specified dtype.

        Notes
        -----
        This wrapper handles the conversion of arrays to NumPy arrays with
        different dtypes.
        """
        return partial(utils.conversion_fn_wrapper, fn=fn, precision=self.precision)

    def accumulate_grads(self, gradient: np.ndarray, input: np.ndarray, cache, idx):
        return utils.accumulate_grads(gradient, input, cache, idx)

    def array(self, data: Any, *, dtype: Dtype | None = None) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._conversion_fn_wrapper(np.array)(
            data, dtype=utils.dtype_map[_dtype]
        )

    def zeros(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(np.zeros)(
            shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def ones(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(np.ones)(
            shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def ones_like(self, input: np.ndarray, *, dtype: Dtype | None = None) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(np.ones_like)(
            input, dtype=utils.dtype_map[_dtype]
        )

    def zeros_like(
        self, input: np.ndarray, *, dtype: Dtype | None = None
    ) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(np.zeros_like)(
            input, dtype=utils.dtype_map[_dtype]
        )

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(np.random.randn)(
            *_shape, dtype=utils.dtype_map[_dtype]
        )

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(np.random.rand)(
            *_shape, dtype=utils.dtype_map[_dtype]
        )

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(np.random.randint)(
            low=low, high=high, size=_shape, dtype=utils.dtype_map[_dtype]
        )

    def rand_uniform(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(np.random.uniform)(
            low=low, high=high, size=_shape, dtype=utils.dtype_map[_dtype]
        )

    def arange(self, *args, dtype: Dtype | None = None) -> np.ndarray:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(np.arange)(
            *args, dtype=utils.dtype_map[_dtype]
        )

    def to_numpy(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def flatten(
        self, input: np.ndarray, start_dim: int = 0, end_dim: int = -1
    ) -> np.ndarray:
        return ops.flatten(input, start_dim=start_dim, end_dim=end_dim)

    def abs(self, input: np.ndarray) -> np.ndarray:
        return np.abs(input)

    def sign(self, input: np.ndarray) -> np.ndarray:
        return np.sign(input)

    def sin(self, input: np.ndarray) -> np.ndarray:
        return np.sin(input)

    def cos(self, input: np.ndarray) -> np.ndarray:
        return np.cos(input)

    def tanh(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    def relu(self, input: np.ndarray) -> np.ndarray:
        return ops.relu(input)

    def leaky_relu(self, input: np.ndarray, slope: float | np.ndarray) -> np.ndarray:
        return ops.leaky_relu(input, slope)

    def sigmoid(self, input: np.ndarray) -> np.ndarray:
        return ops.sigmoid(input)

    def softplus(self, input: np.ndarray) -> np.ndarray:
        return ops.softplus(input)

    def softmax(self, input: np.ndarray, dim: int = -1) -> np.ndarray:
        # TODO: dim can be Sequence[int] as well. Should work
        # for all backends.
        return ops.softmax(input, axis=dim)

    def log(self, input: np.ndarray) -> np.ndarray:
        return np.log(input)

    def isnan(self, input: np.ndarray) -> np.ndarray:
        return np.isnan(input)

    def stop_gradient(self, data: np.ndarray) -> np.ndarray:
        return ops.stop_gradient(data)

    def squeeze(self, input: np.ndarray) -> np.ndarray:
        return np.squeeze(input)

    def reshape(self, input: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(input, shape)

    def sort(
        self, input: np.ndarray, axis: int = -1, descending: bool = False
    ) -> np.ndarray:
        if descending:
            return -np.sort(-input, axis=axis)
        return np.sort(
            input,
            axis=axis,
        )

    def expand_dims(self, input: np.ndarray, axis: int) -> np.ndarray:
        return np.expand_dims(input, axis)

    def stack(self, inputs: list[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.stack(inputs, axis=axis)

    def cat(
        self, inputs: tuple[np.ndarray, ...] | list[np.ndarray], dim: int = 0
    ) -> np.ndarray:
        return ops.concat(*inputs, axis=dim)

    def pad(self, input: np.ndarray, pad_width: PadWidthType) -> np.ndarray:
        return np.pad(input, pad_width)

    def all(self, input: np.ndarray) -> np.ndarray:
        return np.array(np.all(input))

    def any(self, input: np.ndarray) -> np.ndarray:
        return np.array(np.any(input))

    def atleast_1d(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        return np.atleast_1d(*inputs)

    def atleast_2d(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        return np.atleast_2d(*inputs)

    def transpose(
        self, input: np.ndarray, axes: tuple[int, ...] | list[int] | None = None
    ) -> np.ndarray:
        return ops.transpose(input, axes)

    def where(
        self, cond: np.ndarray, input1: np.ndarray, input2: np.ndarray
    ) -> np.ndarray:
        return ops.where(cond, input1, input2)

    # TODO: Analyze the code's efficiency and refactor it if necessary.
    # topk_namedtuple = namedtuple('topk_namedtuple', ['values', 'indices'])
    def topk(self, array: np.ndarray, k: int) -> np.ndarray:
        flat = array.ravel()
        indices = np.argpartition(flat, -k)[-k:]
        argsort = np.argsort(-flat[indices])

        indices = indices[argsort]
        values = flat[indices]
        leading_dims = len(array.shape) - len(values.shape)
        values = values.reshape((-1,) * leading_dims + values.shape)
        return values

    def multinomial(
        self, probs: np.ndarray, num_samples: int, replacement: bool = False, **kwargs
    ) -> np.ndarray:
        # input = np.asarray(probs)
        if probs.ndim == 1:
            probs = probs[None, :]
            squeeze_result = True
        else:
            squeeze_result = False

        batch_size, num_categories = probs.shape

        # Normalize probabilities
        probs = probs / np.sum(probs, axis=-1, keepdims=True)

        if replacement:
            # Use standard numpy.random.choice
            samples = np.vstack(
                [
                    np.random.choice(
                        num_categories, size=num_samples, p=p, replace=True
                    )
                    for p in probs
                ]
            )
        else:
            if num_samples > num_categories:
                raise ValueError(
                    f"Cannot sample {num_samples} samples without replacement "
                    f"from {num_categories} categories"
                )

            # Gumbel-max trick for parallel sampling without replacement
            z = -np.log(-np.log(np.random.random((batch_size, num_categories))))
            # Add log probabilities for Gumbel-max trick
            z = z + np.log(np.maximum(probs, 1e-37))
            # Get top k indices
            samples = np.argsort(-z, axis=1)[:, :num_samples]

        if squeeze_result:
            samples = np.squeeze(samples, axis=0)

        return samples
