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
from typing import Any

import numpy as np

from ....cores.python.numpy import ops, ops_grad
from ....cores.python.numpy import utils as core_utils
from ....types import Dtype
from ...backend import Backend, PadWidthType
from ...utils import StaticScalar, process_shape
from . import utils

CacheType = dict[str, Any]


class NumpyBackend(Backend[np.ndarray[Any, Any]]):
    """A backend implementation for the Mithril library using NumPy with
    manual gradient support.

    Parameters
    ----------
    device: str, optional
        The device on which to perform computations, default is "cpu".
    precision: int, optional
        The precision of the arrays, either 32 or 64, default is 32.
    """

    backend_type = "numpy"

    registered_primitives = {}
    supported_dtypes = [Dtype.float16, Dtype.float32, Dtype.float64]
    primitive_fn_path = "mithril.cores.python.numpy.ops"
    primitive_grad_fn_path = "mithril.cores.python.numpy.ops_grad"
    registered_primitives_grad_fn: dict[str, Callable[..., np.ndarray[Any, Any]]] = {}
    CODEGEN_CONFIG = utils.CODEGEN_CONFIG

    def __init__(self, device: str = "cpu", dtype: Dtype = Dtype.float32) -> None:
        self._dtype = dtype

        if device != "cpu":
            raise RuntimeError(
                f"Specified device: '{device}' is not available!"
                f"Available devices: {NumpyBackend.get_available_devices()}"
            )
        self._device = device

        super().__init__(dtype=dtype)

        self.dtype_map = core_utils.dtype_map
        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict
        self.primitive_grad_function_dict = ops_grad.primitive_grad_func_dict
        self._seed_generator = np.random.default_rng(self.seed)

        for key, value in core_utils.dtype_map.items():
            setattr(self, key, value)

    @property
    def is_manualgrad(self) -> bool:
        return True

    @property
    def inf(self) -> float:
        return np.inf

    @property
    def nan(self) -> float:
        return np.nan

    @property
    def DataType(self) -> type[np.ndarray[Any, Any]]:  # noqa: N802
        return utils.ArrayType

    def get_backend_array_type(self) -> type[np.ndarray[Any, Any]]:
        return np.ndarray

    def get_device(self) -> Any:
        return self._device

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
    def register_primitive(  # type: ignore
        fn: Callable[..., Any], fn_grad: Callable[..., np.ndarray[Any, Any]]
    ) -> None:
        formula_key = fn.__name__
        NumpyBackend.registered_primitives[formula_key] = fn
        NumpyBackend.registered_primitives_grad_fn[formula_key + "_grad"] = fn_grad

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self._seed_generator = np.random.default_rng(seed)

    def accumulate_grads(
        self,
        gradient: np.ndarray[Any, Any],
        input: np.ndarray[Any, Any],
        cache: CacheType | None,
        idx: int,
    ) -> np.ndarray[Any, Any]:
        return core_utils.accumulate_grads(gradient, input, cache, idx)

    def array(self, data: Any, *, dtype: Dtype | None = None) -> np.ndarray[Any, Any]:
        _dtype = utils.determine_dtype(data, dtype, self._dtype, self.precision)

        return np.array(data, dtype=core_utils.dtype_map[_dtype])

    def zeros(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> np.ndarray[Any, Any]:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return np.zeros(_shape, dtype=_dtype)

    def ones(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> np.ndarray[Any, Any]:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return np.ones(_shape, dtype=_dtype)

    def argmax(
        self,
        input: np.ndarray[Any, Any],
        axis: int | None = None,
        keepdim: bool = False,
        cache: CacheType | None = None,
    ) -> np.ndarray[Any, Any]:
        return np.argmax(input, axis=axis, keepdims=keepdim)

    def ones_like(
        self, input: np.ndarray[Any, Any], *, dtype: Dtype | None = None
    ) -> np.ndarray[Any, Any]:
        _dtype = self._process_dtype(dtype)
        return np.ones_like(input, dtype=_dtype)

    def zeros_like(
        self, input: np.ndarray[Any, Any], *, dtype: Dtype | None = None
    ) -> np.ndarray[Any, Any]:
        _dtype = self._process_dtype(dtype)
        return np.zeros_like(input, dtype=_dtype)

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> np.ndarray[Any, Any]:
        self._set_seed(key)
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return np.array(np.random.randn(*_shape), dtype=_dtype)

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> np.ndarray[Any, Any]:
        self._set_seed(key)
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return np.array(np.random.rand(*_shape), dtype=_dtype)

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> np.ndarray[Any, Any]:
        self._set_seed(key)
        _dtype = self._process_dtype(dtype, int)
        _shape = process_shape(shape)
        return np.random.randint(low, high, size=_shape).astype(_dtype)

    def rand_uniform(
        self,
        low: int | float | bool | np.ndarray[Any, Any],
        high: int | float | bool | np.ndarray[Any, Any],
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> np.ndarray[Any, Any]:
        self._set_seed(key)
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return np.array(np.random.uniform(low, high, size=_shape), dtype=_dtype)

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float,
        dtype: Dtype | None = None,
    ) -> np.ndarray[Any, Any]:
        default_type = (
            float if any(isinstance(x, float) for x in (start, stop, step)) else int
        )
        _dtype = self._process_dtype(dtype, default_type)
        return np.arange(start, stop, step, dtype=_dtype)

    def linspace(
        self,
        start: int | float | bool | np.ndarray[Any, Any],
        stop: int | float | bool | np.ndarray[Any, Any],
        steps: int,
        dtype: Dtype | None = None,
    ) -> np.ndarray[Any, Any]:
        _dtype = self._process_dtype(dtype)
        return np.linspace(start, stop, steps, dtype=_dtype)

    def flatten(
        self, input: np.ndarray[Any, Any], start_dim: int = 0, end_dim: int = -1
    ) -> np.ndarray[Any, Any]:
        return ops.flatten(input, start_dim=start_dim, end_dim=end_dim)

    def concat(
        self,
        input: list[np.ndarray[Any, Any]] | tuple[np.ndarray[Any, Any], ...],
        axis: int = 0,
    ) -> np.ndarray[Any, Any]:
        return np.concatenate(input, axis=axis)

    def abs(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.abs(input)

    def sign(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.sign(input)

    def sin(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.sin(input)

    def cos(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.cos(input)

    def tanh(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.tanh(input)

    def relu(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return ops.relu(input)

    def leaky_relu(
        self, input: np.ndarray[Any, Any], slope: float | np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        return ops.leaky_relu(input, slope)

    def sigmoid(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return ops.sigmoid(input)

    def softplus(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return ops.softplus(input)

    def softmax(
        self, input: np.ndarray[Any, Any], dim: int = -1
    ) -> np.ndarray[Any, Any]:
        # TODO: dim can be Sequence[int] as well. Should work
        # for all backends.
        return ops.softmax(input, axis=dim)

    def log(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.log(input)

    def isnan(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.isnan(input)

    def stop_gradient(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return ops.stop_gradient(input)

    def squeeze(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.squeeze(input)

    def reshape(
        self, input: np.ndarray[Any, Any], shape: tuple[int, ...]
    ) -> np.ndarray[Any, Any]:
        return np.reshape(input, shape)

    def sort(
        self, input: np.ndarray[Any, Any], axis: int = -1, descending: bool = False
    ) -> np.ndarray[Any, Any]:
        if descending:
            return -np.sort(-input, axis=axis)
        return np.sort(
            input,
            axis=axis,
        )

    def expand_dims(
        self, input: np.ndarray[Any, Any], axis: int
    ) -> np.ndarray[Any, Any]:
        return np.expand_dims(input, axis)

    def stack(
        self, inputs: list[np.ndarray[Any, Any]], axis: int = 0
    ) -> np.ndarray[Any, Any]:
        return np.stack(inputs, axis=axis)

    def cat(
        self,
        input: list[np.ndarray[Any, Any]],
        axis: int = 0,
    ) -> np.ndarray[Any, Any]:
        return ops.concat(input, axis=axis)

    def pad(
        self, input: np.ndarray[Any, Any], pad_width: PadWidthType
    ) -> np.ndarray[Any, Any]:
        return np.pad(input, pad_width)

    def all(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.array(np.all(input))

    def any(self, input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.array(np.any(input))

    def atleast_1d(
        self, inputs: np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], ...]
    ) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], ...]:
        if isinstance(inputs, tuple):
            return np.atleast_1d(*inputs)
        else:
            return np.atleast_1d(inputs)

    def atleast_2d(
        self, inputs: np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], ...]
    ) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], ...]:
        if isinstance(inputs, tuple):
            return np.atleast_2d(*inputs)
        else:
            return np.atleast_2d(inputs)

    def transpose(
        self,
        input: np.ndarray[Any, Any],
        axes: tuple[int, ...] | list[int] | None = None,
    ) -> np.ndarray[Any, Any]:
        return ops.transpose(input, axes)

    def where(
        self,
        cond: np.ndarray[Any, Any],
        input1: np.ndarray[Any, Any],
        input2: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        return ops.where(cond, input1, input2)

    # TODO: Analyze the code's efficiency and refactor it if necessary.
    # topk_namedtuple = namedtuple('topk_namedtuple', ['values', 'indices'])

    # TODO: Now topk only supports one dimensional tensors,
    # add multi-dimensional support similar to torch, jax and mlx
    def topk(self, input: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
        flat = input.ravel()
        indices = np.argpartition(flat, -k)[-k:]
        argsort = np.argsort(-flat[indices])

        indices = indices[argsort]
        values = flat[indices]
        leading_dims = len(input.shape) - len(values.shape)
        values = values.reshape((-1,) * leading_dims + values.shape)
        return values

    def multinomial(
        self,
        probs: np.ndarray[Any, Any],
        num_samples: int,
        replacement: bool = False,
        key: int | None = None,
    ) -> np.ndarray[Any, Any]:
        self._set_seed(key)
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

    def clip(
        self,
        input: np.ndarray[Any, Any],
        min: np.ndarray[Any, Any] | StaticScalar,
        max: np.ndarray[Any, Any] | StaticScalar,
    ) -> np.ndarray[Any, Any]:
        return np.clip(input, min, max)

    def convert_to_logical(self, input: Any, force: bool = False) -> Any:
        # Try dtype:
        if input.__hash__ and input in core_utils.dtype_map.inverse:
            return Dtype[core_utils.dtype_map.inverse[input]]
        if isinstance(input, np.dtype):
            return Dtype[input.name]

        if force:
            raise ValueError(f"Invalid value '{input}'!")

        return input

    def _process_dtype(
        self,
        dtype: Dtype | None = None,
        default_type: type[float] | type[int] | type[bool] = float,
    ) -> np.dtype[Any]:
        if isinstance(dtype, Dtype):
            return core_utils.dtype_map[dtype.name]
        elif dtype is None:
            return core_utils.dtype_map[default_type.__name__ + str(self.precision)]
        else:
            raise ValueError(f"Invalid dtype {dtype}")

    def _set_seed(self, seed: int | None) -> None:
        _seed = self._seed_generator.integers(0, 2**14) if seed is None else seed
        np.random.seed(_seed)
