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

from ....core import Dtype
from ....cores.mlx import ops
from ....cores.mlx import utils as core_utils
from ...backend import Backend, PadWidthType
from ...utils import DtypeSubTypes, StaticScalar, process_shape
from . import utils

__all__ = ["MlxBackend"]

AxisType = None | int | Sequence[int]


class MlxBackend(Backend[mx.array]):
    backend_type = "mlx"
    supported_dtypes = [Dtype.float16, Dtype.bfloat16, Dtype.float32]
    registered_primitives: dict[str, Callable[..., mx.array]] = {}
    primitive_fn_path = "mithril.cores.mlx.ops"

    def __init__(
        self,
        device: str = "cpu",
        dtype: Dtype = Dtype.float32,
        eager_free: bool = False,
    ) -> None:
        if eager_free:
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        self._dtype = dtype
        self._device = device
        super().__init__(dtype=dtype)

        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict
        self.prng_key = mx.random.key(self.seed)

        for key, value in core_utils.dtype_map.items():
            setattr(self, key, value)

    @property
    def is_manualgrad(self) -> bool:
        return False

    @property
    def inf(self) -> float:
        return mx.inf

    @property
    def nan(self) -> float:
        return mx.nan

    @property
    def device(self) -> Any:
        utils.get_device(self._device)

    @property
    def codegen_config(self) -> dict[str, bool]:
        return utils.CODEGEN_CONFIG

    def get_device(self) -> Any:
        return self._device

    @property
    def DataType(self) -> type[mx.array]:  # noqa: N802
        return utils.ArrayType

    # TODO: This property is weird! Investigate why this property is used.

    def get_backend_array_type(self) -> type[mx.array]:
        return mx.array

    @staticmethod
    def get_available_devices() -> list[str]:
        return utils.get_available_devices()

    @staticmethod
    def register_primitive(fn: Callable[..., mx.array]) -> None:
        MlxBackend.registered_primitives[fn.__name__] = fn

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.prng_key = mx.random.key(seed)

    def argmax(
        self, input: mx.array, axis: AxisType = None, keepdim: bool = False
    ) -> mx.array:
        return mx.argmax(input, axis=axis, keepdims=keepdim)

    def to_device(
        self, data: mx.array, device: str, asynchronous: bool = True
    ) -> mx.array:
        return data

    def block_until_ready(self, data: mx.array) -> None:
        mx.eval(data)

    def _handle_dict_type_fun(
        self,
        *inputs: mx.array,
        keys: list[str],
        cotangent_keys: list[str],
        fn: Callable[..., Any],
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
        fn: Callable[..., Any],
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

    def array(self, input: Any, *, dtype: Dtype | None = None) -> mx.array:
        _dtype = utils.determine_dtype(input, dtype, self._dtype, self.precision)
        return mx.array(input, dtype=core_utils.dtype_map[_dtype])

    def zeros(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> mx.array:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return mx.zeros(_shape, dtype=_dtype)

    def ones(
        self, *shape: int | tuple[int, ...] | list[int], dtype: Dtype | None = None
    ) -> mx.array:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return mx.ones(_shape, dtype=_dtype)

    def ones_like(self, input: mx.array, *, dtype: Dtype | None = None) -> mx.array:
        if dtype is not None:
            raise ValueError("dtype argument is not supported for ones_like")

        return mx.ones_like(input)

    def zeros_like(self, input: mx.array, *, dtype: Dtype | None = None) -> mx.array:
        if dtype is not None:
            raise ValueError("dtype argument is not supported for ones_like")

        return mx.zeros_like(input)

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> mx.array:
        prng_key = self._get_prng_key(key)
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return mx.random.normal(shape=_shape, dtype=_dtype, key=prng_key)

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> mx.array:
        prng_key = self._get_prng_key(key)
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return mx.random.uniform(shape=_shape, dtype=_dtype, key=prng_key)

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> mx.array:
        prng_key = self._get_prng_key(key)
        _dtype = self._process_dtype(dtype, "int")
        _shape = process_shape(shape)
        return mx.random.randint(low, high, shape=_shape, dtype=_dtype, key=prng_key)

    def rand_uniform(
        self,
        low: int | float | bool | mx.array,
        high: int | float | bool | mx.array,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        key: int | None = None,
    ) -> mx.array:
        prng_key = self._get_prng_key(key)
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)
        return mx.random.uniform(low, high, shape=_shape, dtype=_dtype, key=prng_key)

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float,
        dtype: Dtype | None = None,
    ) -> mx.array:
        default_type = (
            "float" if any(isinstance(x, float) for x in (start, stop, step)) else "int"
        )
        _dtype = self._process_dtype(dtype, default_type)

        return mx.arange(start, stop, step, dtype=_dtype)

    def linspace(
        self,
        start: int | float | bool | mx.array,
        stop: int | float | bool | mx.array,
        steps: int,
        dtype: Dtype | None = None,
    ) -> mx.array:
        _dtype = self._process_dtype(dtype)
        return mx.linspace(start, stop, steps, dtype=_dtype)

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

    def stop_gradient(self, input: mx.array) -> mx.array:
        return mx.stop_gradient(input)

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

    def atleast_1d(
        self, inputs: mx.array | tuple[mx.array, ...]
    ) -> mx.array | tuple[mx.array, ...]:
        if isinstance(inputs, tuple):
            return mx.atleast_1d(*inputs)
        else:
            return mx.atleast_1d(inputs)

    def atleast_2d(
        self, inputs: mx.array | tuple[mx.array, ...]
    ) -> mx.array | tuple[mx.array, ...]:
        if isinstance(inputs, tuple):
            return mx.atleast_2d(*inputs)
        else:
            return mx.atleast_2d(inputs)

    def transpose(
        self, input: mx.array, axes: tuple[int, ...] | list[int] | None = None
    ) -> mx.array:
        return ops.transpose(input, axes)

    def where(self, cond: mx.array, input1: mx.array, input2: mx.array) -> mx.array:
        return mx.where(cond, input1, input2)

    def topk(self, input: mx.array, k: int) -> mx.array:
        return -mx.sort(-mx.topk(input, k))

    def multinomial(
        self,
        probs: mx.array,
        num_samples: int,
        replacement: bool = False,
        key: int | None = None,
    ) -> mx.array:
        """
        Faster JAX implementation of multinomial sampling.

        Args:
            key: JAX PRNG key
            input: 1D or 2D array of probabilities
            num_samples: number of samples to draw
            replacement: whether to sample with replacement
        """
        prng_key = self._get_prng_key(key)
        input = mx.array(probs)
        input = input / mx.sum(input, axis=-1, keepdims=True)
        batch_size = input.shape[:-1]
        logits = mx.log(mx.maximum(input, 1e-37))

        if replacement:
            # Use categorical directly - much faster than choice
            samples = mx.random.categorical(
                logits,  # avoid log(0)
                shape=batch_size + (num_samples,),
                key=prng_key,
            )
        else:
            # TODO: This algorithm is not efficient for small num_samples
            # consider more efficient algorithm

            # For without replacement, use Gumbel-max trick
            # This is much faster than using choice
            z = mx.random.gumbel(shape=input.shape + (num_samples,), key=prng_key)  # type: ignore
            # Add log probabilities for Gumbel-max trick,
            z = z + logits[..., None]
            # Get top k indices
            samples = mx.argsort(-z, axis=input.ndim - 1)[..., :num_samples, 0]

        return samples

    def clip(
        self,
        input: mx.array,
        min: mx.array | StaticScalar,
        max: mx.array | StaticScalar,
    ) -> mx.array:
        return mx.clip(input, min, max)

    def jit[**P, T](self, fn: Callable[P, T]) -> Callable[P, T]:
        return fn

    def grad(
        self, fn: Callable[..., dict[str, mx.array]]
    ) -> Callable[..., dict[str, mx.array]]:
        return mx.grad(fn)

    def value_and_grad(
        self, fn: Callable[..., dict[str, mx.array]]
    ) -> Callable[..., tuple[dict[str, mx.array], dict[str, mx.array]]]:
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
    ) -> tuple[Sequence[mx.array], Callable[..., Any], Sequence[mx.array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, mx.array]],
        primals: dict[str, mx.array],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[dict[str, mx.array], Callable[..., Any], dict[str, mx.array]]: ...

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
        dict[str, mx.array] | list[mx.array] | Callable[..., Any],
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
            else:
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

    def vmap(  # type: ignore  #mypy bug
        self, fn: Callable[[mx.array], mx.array]
    ) -> Callable[[mx.array], mx.array]:
        return mx.vmap(fn)

    def _process_dtype(
        self,
        dtype: Dtype | None = None,
        default_type: str | None = None,
    ) -> mx.Dtype:
        if isinstance(dtype, Dtype):
            return core_utils.dtype_map[dtype.name]
        elif dtype is None:
            if default_type is None:
                default_type = self._get_default_subtype()
            return core_utils.dtype_map[default_type + str(self.precision)]
        else:
            raise ValueError(f"Invalid dtype {dtype}")

    def _get_prng_key(self, key: int | None) -> mx.array:
        if key is None:
            _key = self.prng_key
            self.prng_key, _ = mx.random.split(_key)
            return _key
        return mx.random.key(key)

    def _get_default_subtype(self) -> str:
        return DtypeSubTypes[self._dtype.name].value
