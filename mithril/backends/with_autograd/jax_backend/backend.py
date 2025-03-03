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

import math
import os
from collections.abc import Callable, Sequence
from typing import Any, overload

import jax
import jax.numpy as jnp

from ....cores.python.jax import ops
from ....cores.python.jax import utils as core_utils
from ....types import Dtype
from ...backend import PadWidthType, ParallelBackend
from ...utils import DtypeSubTypes, StaticScalar, process_shape
from . import utils
from .parallel import JaxParallel

__all__ = ["JaxBackend"]

jax.config.update("jax_enable_x64", True)  # type: ignore

AxisType = None | int | Sequence[int]


class JaxBackend(ParallelBackend[jax.numpy.ndarray]):
    """JaxBackend: A backend implementation for the Mithril library using Jax.

    Parameters
    ----------
    device: str, optional
        The device on which to perform computations, default is "cpu".
    precision: int, optional
        The precision of the arrays, either 32 or 64, default is 32.
    pre_allocate: bool, optional
        This argument controls whether JAX pre-allocates memory, default is False.
    """

    backend_type = "jax"
    registered_primitives: dict[str, Callable[..., jax.numpy.ndarray]] = {}
    primitive_fn_path = "mithril.cores.python.jax.ops"
    CODEGEN_CONFIG = utils.CODEGEN_CONFIG

    def __init__(
        self,
        device: str = "cpu",
        dtype: Dtype = Dtype.float32,
        pre_allocate: bool = False,
        device_mesh: tuple[int, ...] | None = None,
    ) -> None:
        self._device = device
        core_utils.get_device(device)  # Check device is available
        self._dtype = dtype
        self._parallel_manager: JaxParallel | None = None

        super().__init__(dtype=dtype, device_mesh=device_mesh)

        if device_mesh is not None:
            self._create_parallel(device_mesh=device_mesh)

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(pre_allocate).lower()

        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict
        self.dtype_map = core_utils.dtype_map
        self.prng_key = jax.random.PRNGKey(self.seed)

        for key, value in core_utils.dtype_map.items():
            setattr(self, key, value)

    @property
    def is_manualgrad(self) -> bool:
        return False

    @property
    def inf(self) -> float:
        return jax.numpy.inf

    @property
    def nan(self) -> float:
        return jax.numpy.nan

    def get_backend_array_type(self) -> type[jax.Array]:
        return jax.Array

    @property
    def device(self) -> jax.Device:
        return core_utils.get_device(self._device)

    def get_device(self) -> Any:
        return self._device

    # TODO: This property is weird! Investigate why this property is used.
    @property
    def DataType(self) -> type[jax.Array]:  # noqa: N802
        return utils.ArrayType

    @staticmethod
    def get_available_devices() -> list[str]:
        """Static method to get a list of available devices.

        Parameters
        ----------
        list[str]
            List of available devices.
        """
        return core_utils.get_available_devices()

    @staticmethod
    def register_primitive(fn: Callable[..., Any]) -> None:
        JaxBackend.registered_primitives[fn.__name__] = fn

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.prng_key = jax.random.PRNGKey(seed)

    def to_device(
        self, data: jax.Array, device: str, asynchronous: bool = True
    ) -> jax.Array:
        """Move data to the specified device.

        Parameters
        ----------
        data: jax.Array
            The data to be moved to the specified device.
        device: str
            The target device for the data.
        """
        _device = core_utils.get_device(device)
        if not asynchronous:
            return jax.device_put(data, device=_device).block_until_ready()
        return jax.device_put(data, device=_device)

    def block_until_ready(self, data: jax.Array) -> jax.Array | None:
        """Block until the specified data is ready.

        Parameters
        ----------
        data: jax.Array
            The data for which the method will block until it is ready.
        """
        return data.block_until_ready()

    def register_callable(
        self, fn: Callable[..., Any], fn_name: str, jit: bool = False
    ) -> None:
        assert (
            self._parallel_manager is not None
        ), "Parallel manager is not initialized!"

        fn_name = str(id(self)) + fn_name
        return self._parallel_manager.register_callable(fn, fn_name, jit)

    def _run_callable(self, *primals: jax.Array, fn_name: str) -> Any:
        assert (
            self._parallel_manager is not None
        ), "Parallel manager is not initialized!"

        fn_name = str(id(self)) + fn_name
        return self._parallel_manager.run_callable(*primals, fn_name=fn_name)

    def _create_parallel(self, device_mesh: tuple[int, ...]) -> None:
        self._parallel_manager = JaxParallel(math.prod(device_mesh), self._device)

    def array(
        self,
        input: Any,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype = utils.determine_dtype(input, dtype, self._dtype, self.precision)

        with jax.default_device(self.device):
            array = jax.numpy.array(input, dtype=core_utils.dtype_map[_dtype])

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def zeros(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        with jax.default_device(self.device):
            array = jax.numpy.zeros(_shape, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def ones(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        with jax.default_device(self.device):
            array = jax.numpy.ones(_shape, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def ones_like(
        self,
        input: jax.Array,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype = self._process_dtype(dtype) if dtype is not None else None

        with jax.default_device(self.device):
            array = jax.numpy.ones_like(input, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def zeros_like(
        self,
        input: jax.Array,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype = self._process_dtype(dtype) if dtype is not None else None

        with jax.default_device(self.device):
            array = jax.numpy.zeros_like(input, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> jax.Array:
        prng_key = self._get_prng_key(key)

        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        with jax.default_device(self.device):
            array = jax.random.normal(prng_key, _shape, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> jax.Array:
        prng_key = self._get_prng_key(key)

        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        with jax.default_device(self.device):
            array = jax.random.normal(prng_key, _shape, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> jax.Array:
        prng_key = self._get_prng_key(key)

        _dtype = self._process_dtype(dtype, "int")
        _shape = process_shape(shape)

        with jax.default_device(self.device):
            array = jax.random.randint(prng_key, _shape, low, high, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def rand_uniform(
        self,
        low: int | float | bool | jax.numpy.ndarray,
        high: int | float | bool | jax.numpy.ndarray,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> jax.Array:
        prng_key = self._get_prng_key(key)

        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        with jax.default_device(self.device):
            array = jax.random.uniform(prng_key, _shape, _dtype, low, high)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        default_type = (
            "float" if any(isinstance(x, float) for x in (start, stop, step)) else "int"
        )
        _dtype = self._process_dtype(dtype, default_type)

        with jax.default_device(self.device):
            array = jax.numpy.arange(start, stop, step, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def linspace(
        self,
        start: int | float | bool | jax.numpy.ndarray,
        stop: int | float | bool | jax.numpy.ndarray,
        steps: int,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype = self._process_dtype(dtype)
        with jax.default_device(self.device):
            array = jax.numpy.linspace(start, stop, steps, dtype=_dtype)

        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(array, device_mesh)

        return array

    def flatten(
        self, input: jax.Array, start_dim: int = 0, end_dim: int = -1
    ) -> jax.Array:
        return ops.flatten(input, start_dim=start_dim, end_dim=end_dim)

    def concat(
        self, inputs: tuple[jax.Array, ...] | list[jax.Array], axis: int = 0
    ) -> jax.Array:
        return jax.numpy.concat(inputs, axis=axis)

    def abs(self, input: jax.Array) -> jax.Array:
        return jax.numpy.abs(input)

    def sign(self, input: jax.Array) -> jax.Array:
        return jax.numpy.sign(input)

    def sin(self, input: jax.Array) -> jax.Array:
        return jax.numpy.sin(input)

    def cos(self, input: jax.Array) -> jax.Array:
        return jax.numpy.cos(input)

    def tanh(self, input: jax.Array) -> jax.Array:
        return jax.nn.tanh(input)

    def relu(self, input: jax.Array) -> jax.Array:
        return jax.nn.relu(input)

    def leaky_relu(self, input: jax.Array, slope: float | jax.Array) -> jax.Array:
        return jax.nn.leaky_relu(input, slope)

    def sigmoid(self, input: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(input)

    def softplus(self, input: jax.Array) -> jax.Array:
        return jax.nn.softplus(input)

    def softmax(self, input: jax.Array, dim: int = -1) -> jax.Array:
        # TODO: dim can be Sequence[int] as well. Should work
        # for all backends.
        return ops.softmax(input, axis=dim)

    def argmax(
        self, input: jax.Array, axis: int | None = None, keepdim: bool = False
    ) -> jax.Array:
        return jnp.argmax(input, axis=axis, keepdims=keepdim)

    def log(self, input: jax.Array) -> jax.Array:
        return jax.numpy.log(input)

    def isnan(self, input: jax.Array) -> jax.Array:
        return jax.numpy.isnan(input)

    def stop_gradient(self, input: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient(input)

    def squeeze(self, input: jax.Array) -> jax.Array:
        return jax.numpy.squeeze(input)

    def reshape(self, input: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jax.numpy.reshape(input, shape)

    def sort(
        self, input: jax.Array, axis: int = -1, descending: bool = False
    ) -> jax.Array:
        return jax.numpy.sort(input, axis, descending=descending)

    def expand_dims(self, input: jax.Array, axis: int) -> jax.Array:
        return jax.numpy.expand_dims(input, axis)

    def stack(self, inputs: list[jax.Array], axis: int = 0) -> jax.Array:
        return jax.numpy.stack(inputs, axis=axis)

    def cat(
        self, inputs: tuple[jax.Array, ...] | list[jax.Array], axis: int = 0
    ) -> jax.Array:
        return ops.concat(*inputs, axis=axis)

    def pad(self, input: jax.Array, pad_width: PadWidthType) -> jax.Array:
        return jax.numpy.pad(input, pad_width)

    def all(self, input: jax.Array) -> jax.Array:
        return jax.numpy.all(input)

    def any(self, input: jax.Array) -> jax.Array:
        return jax.numpy.any(input)

    def atleast_1d(
        self, inputs: jax.Array | tuple[jax.Array, ...]
    ) -> jax.Array | list[jax.Array]:
        if isinstance(inputs, tuple):
            return jax.numpy.atleast_1d(*inputs)
        else:
            return jax.numpy.atleast_1d(inputs)

    def atleast_2d(
        self, inputs: jax.Array | tuple[jax.Array, ...]
    ) -> jax.Array | list[jax.Array]:
        if isinstance(inputs, tuple):
            return jax.numpy.atleast_2d(*inputs)
        else:
            return jax.numpy.atleast_2d(inputs)

    def transpose(
        self, input: jax.Array, axes: tuple[int, ...] | list[int] | None = None
    ) -> jax.Array:
        return input.transpose(axes)

    def unique(
        self, input: jax.Array, **kwargs: Any
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
        return jax.numpy.unique(input, **kwargs)

    def where(self, cond: jax.Array, input1: jax.Array, input2: jax.Array) -> jax.Array:
        return ops.where(cond, input1, input2)

    def topk(self, input: jax.Array, k: int) -> jax.Array:
        return jax.lax.top_k(input, k)[0]

    def multinomial(
        self,
        probs: jax.Array,
        num_samples: int,
        replacement: bool = False,
        key: int | None = None,
    ) -> jax.Array:
        """
        Faster JAX implementation of multinomial sampling.

        Args:
            key: JAX PRNG key
            input: 1D or 2D array of probabilities
            num_samples: number of samples to draw
            replacement: whether to sample with replacement
        """
        prng_key = self._get_prng_key(key)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        logits = jnp.log(probs)

        if replacement:
            # Use categorical directly - much faster than choice
            samples = jax.random.categorical(
                prng_key,
                logits,
                shape=probs.shape[:-1] + (num_samples,),
            )
        else:
            # TODO: This algorithm is not efficient for small num_samples
            # consider more efficient algorithm

            # For without replacement, use Gumbel-max trick
            # This is much faster than using choice
            z = jax.random.gumbel(prng_key, shape=probs.shape + (num_samples,))
            # Add log probabilities for Gumbel-max trick,
            z = z + logits[..., None]
            # Get top k indices
            samples = jax.numpy.argsort(-z, axis=probs.ndim - 1)[..., :num_samples, 0]

        return samples

    def clip(
        self,
        input: jax.Array,
        min: jax.Array | StaticScalar,
        max: jax.Array | StaticScalar,
    ) -> jax.Array:
        return input.clip(min=min, max=max)

    def jit(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> Callable[..., jax.Array | tuple[jax.Array, ...]] | dict[str, jax.Array]:
        return jax.jit(*args, **kwargs)

    def grad(
        self, fn: Callable[..., dict[str, jax.Array]]
    ) -> Callable[..., dict[str, jax.Array]]:
        return jax.grad(fn)

    def value_and_grad(
        self, fn: Callable[..., dict[str, jax.Array]]
    ) -> Callable[..., tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
        return jax.value_and_grad(fn)

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[Sequence[jax.Array], Sequence[jax.Array]]],
        primals: list[jax.Array],
        *,
        cotangents: tuple[jax.Array, ...],
        has_aux: bool = True,
    ) -> tuple[Sequence[jax.Array], list[jax.Array], Sequence[jax.Array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[dict[str, jax.Array], dict[str, jax.Array]]],
        primals: dict[str, jax.Array],
        *,
        cotangents: dict[str, jax.Array],
        has_aux: bool = True,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[jax.Array]],
        primals: list[jax.Array],
        *,
        cotangents: tuple[jax.Array, ...],
        has_aux: bool = False,
    ) -> tuple[Sequence[jax.Array], list[jax.Array], Sequence[jax.Array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, jax.Array]],
        primals: dict[str, jax.Array],
        *,
        cotangents: dict[str, jax.Array],
        has_aux: bool = False,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[jax.Array]],
        primals: list[jax.Array],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[Sequence[jax.Array], Callable[..., Any], Sequence[jax.Array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, jax.Array]],
        primals: dict[str, jax.Array],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[dict[str, jax.Array], Callable[..., Any], dict[str, jax.Array]]: ...

    def vjp(
        self,
        fn: Callable[
            ...,
            dict[str, jax.Array]
            | Sequence[jax.Array]
            | tuple[Sequence[jax.Array], Sequence[jax.Array]]
            | tuple[dict[str, jax.Array], dict[str, jax.Array]],
        ],
        primals: dict[str, jax.Array] | list[jax.Array],
        *,
        cotangents: dict[str, jax.Array] | tuple[jax.Array, ...] | None = None,
        has_aux: bool = False,
    ) -> tuple[
        dict[str, jax.Array] | Sequence[jax.Array] | jax.Array,
        dict[str, jax.Array] | list[jax.Array] | Callable[..., Any],
        dict[str, jax.Array] | Sequence[jax.Array] | jax.Array,
    ]:
        _primals: (
            list[jax.Array | dict[str, jax.Array]] | dict[str, jax.Array] | jax.Array
        ) = primals  # type: ignore
        if isinstance(primals, dict | jax.Array):
            _primals = [primals]
        output, vjp, *aux = jax.vjp(fn, *_primals, has_aux=has_aux)  # type: ignore
        if has_aux:
            (aux,) = aux
        else:
            aux = {} if isinstance(cotangents, dict) else []
        if cotangents is not None:
            vjp = vjp(cotangents)
            if isinstance(cotangents, dict):
                # JAX vjp returns tuple[dict] for dict type returns.
                # So we should unpack vjp result.
                (vjp,) = vjp
        return output, vjp, aux

    def vmap(  # type: ignore # mypy bug
        self, fn: Callable[..., dict[str, jax.Array]]
    ) -> Callable[..., dict[str, jax.Array]]:
        return jax.vmap(fn)

    def jacrev(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return jax.jacrev(fn)

    def jacfwd(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return jax.jacfwd(fn)

    def convert_to_logical(self, input: Any, force: bool = False) -> Any:
        # Try dtype:
        if input.__hash__ and input in core_utils.dtype_map.inverse:
            return Dtype[core_utils.dtype_map.inverse[input]]
        elif isinstance(input, jax.numpy.dtype):
            return Dtype[input.name]

        if force:
            raise ValueError(f"Invalid value '{input}'!")

        return input

    def _process_dtype(
        self,
        dtype: Dtype | None = None,
        default_type: str | None = None,
    ) -> jax.numpy.dtype[Any]:
        if isinstance(dtype, Dtype):
            return core_utils.dtype_map[dtype.name]
        elif dtype is None:
            if default_type is None:
                default_type = self._get_default_subtype()
            return core_utils.dtype_map[default_type + str(self.precision)]
        else:
            raise ValueError(f"Invalid dtype {dtype}")

    def _get_prng_key(self, key: int | None = None) -> jax.Array:
        if key is None:
            _key = self.prng_key
            self.prng_key, _ = jax.random.split(_key)
            return _key
        return jax.random.PRNGKey(key)

    def _get_defualt_type(self) -> jax.numpy.dtype[Any]:
        return getattr(self, self._dtype.name)

    def _get_default_subtype(self) -> str:
        return DtypeSubTypes[self._dtype.name].value
