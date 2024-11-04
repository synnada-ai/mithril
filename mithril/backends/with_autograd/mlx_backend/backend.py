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

import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, overload

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ....core import Dtype
from ...backend import Backend, PadWidthType
from ...utils import process_shape
from . import ops, utils

__all__ = ["MlxBackend"]


class MlxBackend(Backend[mx.array]):
    type = "mlx"
    supported_precisions = [16, 32]
    registered_primitives = {}
    primitive_fn_path = "mithril.backends.with_autograd.mlx_backend.ops"

    def __init__(
        self, device: str = "cpu", precision: int = 32, eager_free: bool = False
    ) -> None:
        if eager_free:
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        self._precision = precision
        self._device = device
        super().__init__()

        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict
        mx.random.seed(self.seed)

    @property
    def is_manualgrad(self):
        return False

    @property
    def inf(self):
        return mx.inf

    @property
    def device(self):
        utils.get_device(self._device)

    @property
    def DataType(self):  # noqa: N802
        return utils.ArrayType

    # TODO: This property is weird! Investigate why this property is used.

    def get_backend_array_type(self):
        return mx.array

    @staticmethod
    def get_available_devices():
        return utils.get_available_devices()

    @staticmethod
    def register_primitive(fn: Callable) -> None:
        MlxBackend.registered_primitives[fn.__name__] = fn

    def set_seed(self, seed: int):
        self.seed = seed
        mx.random.seed(seed)

    def to_device(
        self, data: mx.array, device: str, asynchronous: bool = True
    ) -> mx.array:
        return data

    def block_until_ready(self, data: mx.array):
        mx.eval(data)

    def _creation_fn_wrapper(self, fn: Callable) -> Callable:
        return partial(
            utils.creation_fn_wrapper,
            fn=fn,
            precision=self.precision,
        )

    def _conversion_fn_wrapper(self, fn: Callable) -> Callable:
        return partial(
            utils.conversion_fn_wrapper,
            fn=fn,
            precision=self.precision,
        )

    def _handle_dict_type_fun(
        self,
        *inputs: mx.array,
        keys: list[str],
        cotangent_keys: list[str],
        fn: Callable,
        output_keys: list[str],
        has_aux: bool,
    ) -> list[mx.array]:
        # Used for MLX to convert inputs to list
        input_dict = {}
        for key, input in zip(keys, inputs, strict=False):
            input_dict[key] = input

        _output = fn(input_dict)
        if has_aux:
            # This means function has auxilary outputs
            output, aux = _output
            # In case function has auxilary outputs,
            # cotangent keys must include all output keys.
            if list(output.keys()) != cotangent_keys:
                raise KeyError(
                    "Output keys must match with cotangent keys when has_aux = True"
                )
        else:
            output = _output
            aux = dict()
        output_keys += list(output.keys()) + list(aux.keys())
        return [output[key] for key in cotangent_keys] + [
            self.stop_gradient(value) for value in aux.values()
        ]

    def _handle_sequence_type_fun(
        self,
        *inputs: mx.array,
        cotangents: Sequence[mx.array] | mx.array,
        fn: Callable,
        has_aux: bool,
    ) -> list[mx.array]:
        _output = fn(*inputs)

        if has_aux:
            # This means function has auxilary outputs
            output, aux = _output
            # In case function has auxilary outputs,
            # length of cotangent must be consistent with output.
            if (
                isinstance(output, mx.array)
                and not isinstance(cotangents, mx.array)
                or isinstance(output, Sequence)
                and (
                    not isinstance(cotangents, Sequence)
                    or len(output) != len(cotangents)
                )
            ):
                raise KeyError(
                    "Output type and length must match with cotangent type and \
                    length when has_aux = True"
                )

            if isinstance(aux, mx.array):
                aux = [aux]
            elif isinstance(aux, tuple):
                aux = list(aux)
        else:
            output = _output
            aux = list()

        if isinstance(output, Sequence):
            output = list(output)
            return [
                value if idx < len(cotangents) else self.stop_gradient(value)
                for idx, value in enumerate(output + aux)
            ]
        return [output]

    def array(self, data: Any, *, dtype: Dtype | None = None) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._conversion_fn_wrapper(mx.array)(
            data, dtype=utils.dtype_map[_dtype]
        )

    def zeros(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(mx.zeros)(
            shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def ones(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(mx.ones)(
            shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def ones_like(self, input: mx.array, *, dtype: Dtype | None = None) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(mx.ones_like)(
            input, dtype=utils.dtype_map[_dtype]
        )

    def zeros_like(self, input: mx.array, *, dtype: Dtype | None = None) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(mx.zeros_like)(
            input, dtype=utils.dtype_map[_dtype]
        )

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(mx.random.normal)(
            shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(mx.random.uniform)(
            shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(mx.random.randint)(
            low=low, high=high, shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def rand_uniform(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        prng_key: Any = None,
    ) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(mx.random.uniform)(
            low=low, high=high, shape=_shape, dtype=utils.dtype_map[_dtype]
        )

    def arange(self, *args, dtype: Dtype | None = None) -> mx.array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(mx.arange)(
            *args, dtype=utils.dtype_map[_dtype]
        )

    def to_numpy(self, arr: mx.array) -> np.ndarray:
        return np.array(arr)

    def flatten(
        self, input: mx.array, start_dim: int = 0, end_dim: int = -1
    ) -> mx.array:
        return mx.flatten(input, start_axis=start_dim, end_axis=end_dim)

    def abs(self, input: mx.array) -> mx.array:
        return mx.abs(input)

    def sign(self, input: mx.array) -> mx.array:
        return mx.sign(input)

    def sin(self, input: mx.array) -> mx.array:
        return mx.sin(input)

    def cos(self, input: mx.array) -> mx.array:
        return mx.cos(input)

    def tanh(self, input: mx.array) -> mx.array:
        return mx.tanh(input)

    def relu(self, input: mx.array) -> mx.array:
        return nn.relu(input)

    def leaky_relu(self, input: mx.array, slope: float | mx.array) -> mx.array:
        return nn.leaky_relu(input, slope)

    def sigmoid(self, input: mx.array) -> mx.array:
        return nn.sigmoid(input)

    def softplus(self, input: mx.array) -> mx.array:
        return nn.softplus(input)

    def softmax(self, input: mx.array, dim: int = -1) -> mx.array:
        # TODO: dim can be Sequence[int] as well. Should work
        # for all backends.
        return nn.softmax(input, axis=dim)

    def log(self, input: mx.array) -> mx.array:
        return mx.log(input)

    def isnan(self, input: mx.array) -> mx.array:
        return mx.isnan(input)

    def stop_gradient(self, data: mx.array) -> mx.array:
        return mx.stop_gradient(data)

    def squeeze(self, input: mx.array) -> mx.array:
        return mx.squeeze(input)

    def reshape(self, input: mx.array, shape: tuple[int, ...]) -> mx.array:
        return mx.reshape(input, shape)

    def sort(
        self, input: mx.array, axis: int = -1, descending: bool = False
    ) -> mx.array:
        result = mx.sort(input, axis)
        if descending:
            result = -result
        return result

    def expand_dims(self, input: mx.array, axis: int) -> mx.array:
        return mx.expand_dims(input, axis)

    def stack(self, inputs: list[mx.array], axis: int = 0) -> mx.array:
        return mx.stack(inputs, axis=axis)

    def cat(
        self, inputs: tuple[mx.array, ...] | list[mx.array], axis: int = 0
    ) -> mx.array:
        inputs = list(inputs)
        return mx.concatenate(inputs, axis=axis)

    def pad(self, input: mx.array, pad_width: PadWidthType) -> mx.array:
        return mx.pad(input, pad_width)

    def all(self, input: mx.array) -> mx.array:
        return mx.all(input)

    def any(self, input: mx.array) -> mx.array:
        return mx.any(input)

    def atleast_1d(self, *inputs: mx.array) -> mx.array | tuple[mx.array, ...]:
        return mx.atleast_1d(*inputs)

    def atleast_2d(self, *inputs: mx.array) -> mx.array | tuple[mx.array, ...]:
        return mx.atleast_2d(*inputs)

    def transpose(
        self, input: mx.array, axes: tuple[int, ...] | list[int] | None = None
    ) -> mx.array:
        return ops.transpose(input, axes)

    def where(self, cond: mx.array, input1: mx.array, input2: mx.array) -> mx.array:
        return mx.where(cond, input1, input2)

    def topk(self, input: mx.array, k: int) -> mx.array:
        return -mx.sort(-mx.topk(input, k))

    def multinomial(
        self, probs: mx.array, num_samples: int, replacement: bool = False, **kwargs
    ) -> mx.array:
        """
        MLX implementation matching torch.multinomial behavior.

        Args:
            probs: 1D or 2D array of probabilities.
                If 2D, each row is a distribution
            num_samples: number of samples to draw
            replacement: whether to sample with replacement
            seed: random seed

        Returns:
            1D or 2D array of indices sampled according to probs
        """
        probs = mx.array(probs)
        if probs.ndim == 1:
            probs = probs[None, :]  # Add batch dimension
            squeeze_result = True
        else:
            squeeze_result = False

        batch_size, num_categories = probs.shape

        # Handle zero probabilities like PyTorch
        zeros_mask = probs == 0
        probs = mx.where(zeros_mask, 0, probs)

        # Check if any row has all zeros
        valid_probs = mx.any(probs > 0, axis=1)

        # Normalize probabilities
        probs = probs / mx.maximum(mx.sum(probs, axis=1, keepdims=True), 1e-10)

        if replacement:
            # Generate uniform random numbers
            u = mx.random.uniform(shape=(batch_size, num_samples, 1))

            # Expand probs for comparison with random numbers
            expanded_probs = mx.expand_dims(probs, 1)  # [batch, 1, num_categories]
            cumsum = mx.cumsum(expanded_probs, axis=-1)  # [batch, 1, num_categories]

            # Compare random numbers with cumulative probabilities
            expanded_u = mx.broadcast_to(u, (batch_size, num_samples, num_categories))
            expanded_cumsum = mx.broadcast_to(
                cumsum, (batch_size, num_samples, num_categories)
            )

            # Count how many cumsum values are less than each random number
            samples = mx.sum(expanded_u > expanded_cumsum, axis=-1)

            # Handle invalid probability rows
            samples = mx.where(
                mx.expand_dims(valid_probs, -1), samples, mx.zeros_like(samples)
            )
        else:
            if num_samples > num_categories:
                raise ValueError(
                    f"Cannot sample {num_samples} samples without replacement "
                    f"from {num_categories} categories"
                )

            samples = mx.zeros((batch_size, num_samples), dtype=mx.int32)

            for b in range(batch_size):
                if not valid_probs[b]:
                    continue

                # Generate ordered random values for this batch
                ordered_u = mx.sort(mx.random.uniform(shape=(num_categories,)))

                # Convert probabilities to cumulative sum
                p = probs[b]
                cumsum = mx.cumsum(p)

                # Track used indices to avoid replacement
                used_mask = mx.zeros((num_categories,), dtype=mx.bool_)  # type: ignore
                batch_samples = mx.zeros((num_samples,), dtype=mx.int32)

                for i in range(num_samples):
                    u = ordered_u[i]

                    # Find index considering already used indices
                    valid_cumsum = mx.where(used_mask, 2.0, cumsum)
                    idx = mx.sum(u > valid_cumsum)

                    # Update used mask and store result
                    used_mask = mx.where(
                        mx.arange(num_categories) == idx, True, used_mask
                    )
                    batch_samples = mx.where(
                        mx.arange(num_samples) == i, idx, batch_samples
                    )

                # Update the samples array for this batch

                samples = mx.where(
                    mx.expand_dims(mx.arange(batch_size) == b, -1),  # type: ignore
                    mx.expand_dims(batch_samples, 0),
                    samples,
                )

        if squeeze_result:
            samples = mx.squeeze(samples, axis=0)

        return samples

    def jit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    def grad(self, fn: Callable) -> Callable:
        return mx.grad(fn)

    def value_and_grad(self, fn: Callable) -> Callable:
        return mx.value_and_grad(fn)

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[Sequence[mx.array], Sequence[mx.array]]],
        primals: list[mx.array],
        *,
        cotangents: tuple[mx.array, ...],
        has_aux: bool = True,
    ) -> tuple[Sequence[mx.array], list[mx.array], Sequence[mx.array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[dict[str, mx.array], dict[str, mx.array]]],
        primals: dict[str, mx.array],
        *,
        cotangents: dict[str, mx.array],
        has_aux: bool = True,
    ) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, mx.array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[mx.array]],
        primals: list[mx.array],
        *,
        cotangents: tuple[mx.array, ...],
        has_aux: bool = False,
    ) -> tuple[Sequence[mx.array], list[mx.array], Sequence[mx.array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, mx.array]],
        primals: dict[str, mx.array],
        *,
        cotangents: dict[str, mx.array],
        has_aux: bool = False,
    ) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, mx.array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[mx.array]],
        primals: list[mx.array],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[Sequence[mx.array], Callable, Sequence[mx.array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, mx.array]],
        primals: dict[str, mx.array],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[dict[str, mx.array], Callable, dict[str, mx.array]]: ...

    def vjp(
        self,
        fn: Callable[
            ...,
            dict[str, mx.array]
            | Sequence[mx.array]
            | tuple[Sequence[mx.array], Sequence[mx.array]]
            | tuple[dict[str, mx.array], dict[str, mx.array]],
        ],
        primals: dict[str, mx.array] | list[mx.array],
        *,
        cotangents: dict[str, mx.array] | tuple[mx.array, ...] | None = None,
        has_aux: bool = False,
    ) -> tuple[
        dict[str, mx.array] | Sequence[mx.array] | mx.array,
        dict[str, mx.array] | list[mx.array] | Callable,
        dict[str, mx.array] | Sequence[mx.array] | mx.array,
    ]:
        if cotangents is None:
            raise ValueError("VJP with None cotangents is not supported.")
        _cotangents: list[mx.array] | tuple[mx.array, ...]
        output: dict[str, mx.array] | list[mx.array] | mx.array
        aux: dict[str, mx.array] | list[mx.array] | mx.array
        vjp: dict[str, mx.array] | list[mx.array]
        # This method handles both dict and list type arguments and return.
        if isinstance(primals, dict):
            assert isinstance(cotangents, dict)
            keys: list[str] = []
            _fn = partial(
                self._handle_dict_type_fun,
                keys=list(primals.keys()),
                cotangent_keys=list(cotangents.keys()),
                fn=fn,
                output_keys=keys,
                has_aux=has_aux,
            )
            _primals = list(primals.values())
            _cotangents = list(cotangents.values())
        else:
            assert isinstance(cotangents, Sequence | mx.array)
            _fn = partial(
                self._handle_sequence_type_fun,
                cotangents=cotangents,
                fn=fn,
                has_aux=has_aux,
            )
            _primals = primals
            _cotangents = cotangents
            if isinstance(cotangents, mx.array):
                _cotangents = [cotangents]
            elif isinstance(cotangents, tuple):
                _cotangents = list(cotangents)
        # Calculate VJP.
        out_list, vjp_list = mx.vjp(_fn, _primals, _cotangents)

        if isinstance(primals, dict):
            # Revert to dict representation
            output = {
                key: value
                for key, value in zip(
                    keys[: len(cotangents)], out_list[: len(cotangents)], strict=False
                )
            }
            aux = {
                key: value
                for key, value in zip(
                    keys[len(cotangents) :], out_list[len(cotangents) :], strict=False
                )
            }
            vjp = {
                key: value
                for key, value in zip(list(primals.keys()), vjp_list, strict=False)
            }
        else:
            # Preserve original output type if it is a single array.
            if isinstance(cotangents, mx.array):
                output = out_list[0]
                aux = []
            else:
                output = out_list[: len(cotangents)]
                aux = out_list[len(cotangents) :]
                # Preserve original aux type if it is a single array.
                aux = aux[0] if len(aux) == 1 else aux
            vjp = vjp_list

        return output, vjp, aux

    def vmap(self, fn: Callable) -> Callable:
        return mx.vmap(fn)
