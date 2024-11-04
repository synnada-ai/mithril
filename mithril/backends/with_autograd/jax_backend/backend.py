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
from functools import partial
from typing import Any, overload

import jax
import numpy as np

from ....core import Dtype
from ...backend import PadWidthType, ParallelBackend
from ...utils import process_shape
from . import ops, utils
from .parallel import JaxParallel

__all__ = ["JaxBackend"]

jax.config.update("jax_enable_x64", True)


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

    type = "jax"
    registered_primitives = {}
    primitive_fn_path = "mithril.backends.with_autograd.jax_backend.ops"

    def __init__(
        self,
        device: str = "cpu",
        precision: int = 32,
        pre_allocate: bool = False,
        device_mesh: tuple[int, ...] | None = None,
    ) -> None:
        self._device = device
        utils.get_device(device)  # Check device is available
        self._precision = precision
        self._parallel_manager: JaxParallel | None = None

        super().__init__(device_mesh=device_mesh)

        if device_mesh is not None:
            self._create_parallel(device_mesh=device_mesh)

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(pre_allocate).lower()

        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict
        self.prng_key = jax.random.PRNGKey(self.seed)

    @property
    def is_manualgrad(self):
        return False

    @property
    def inf(self):
        return jax.numpy.inf

    def get_backend_array_type(self):
        return jax.Array

    @property
    def device(self):
        return utils.get_device(self._device)

    # TODO: This property is weird! Investigate why this property is used.
    @property
    def DataType(self):  # noqa: N802
        return utils.ArrayType

    @staticmethod
    def get_available_devices():
        """Static method to get a list of available devices.

        Parameters
        ----------
        list[str]
            List of available devices.
        """
        return utils.get_available_devices()

    @staticmethod
    def register_primitive(fn: Callable) -> None:
        JaxBackend.registered_primitives[fn.__name__] = fn

    def set_seed(self, seed: int):
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
        _device = utils.get_device(device)
        if not asynchronous:
            return jax.device_put(data, device=_device).block_until_ready()
        return jax.device_put(data, device=_device)

    def block_until_ready(self, data: jax.Array):
        """Block until the specified data is ready.

        Parameters
        ----------
        data: jax.Array
            The data for which the method will block until it is ready.
        """
        data.block_until_ready()

    def _creation_fn_wrapper(self, fn: Callable) -> Callable:
        """
        Wrapper for array creation functions.

        Parameters
        ----------
        fn: Callable
            The original array creation function.

        Returns
        -------
        Callable
            A wrapped function that creates arrays with specified dtype and device.

        Notes
        -----
        Ensures that arrays are created with the correct dtype and device.
        """

        array_conversion_fn = partial(
            utils.creation_fn_wrapper,
            fn=fn,
            device=self._device,
            precision=self.precision,
        )
        array_conversion_fn = partial(self._parallelize, fn=array_conversion_fn)

        return array_conversion_fn

    def _conversion_fn_wrapper(self, fn: Callable) -> Callable:
        """
        Wrapper for array conversion functions.

        Parameters
        ----------
        fn: Callable
            The original array conversion function.

        Returns
        -------
        Callable
            A wrapped function that converts arrays with specified dtype and device.

        Notes
        -----
        Handles the conversion of arrays between different dtypes and devices.

        If dtype is provided, it uses `utils._handle_dtype` to ensure a valid dtype.
        If the input data is a JAX Array, it ensures it's on the specified device.
        If dtype is not provided, uses the default device and handles data precision.
        """
        array_conversion_fn = partial(
            utils.conversion_fn_wrapper,
            fn=fn,
            device=self._device,
            precision=self.precision,
        )
        array_conversion_fn = partial(self._parallelize, fn=array_conversion_fn)

        return array_conversion_fn

    def _parallelize(self, *args, fn: Callable, device_mesh, **kwargs) -> jax.Array:
        """
        Parallelizes the function's return tensor across devices.

        Parameters
        ----------
        fn : Callable
            The function whose return tensor will be parallelized.

        device_mesh : tuple[int, ...], optional
            A tuple specifying the device mesh for parallelization.
            If not provided, the default device mesh is used.

        Returns
        -------
        Callable
            Return tensor parallelized across the specified device mesh.
        """

        tensor: jax.Array = fn(*args, **kwargs)
        if self._parallel_manager is None:
            return tensor
        return self._parallel_manager.parallelize(tensor, device_mesh)

    def _register_callable(
        self, fn: Callable | partial, fn_name: str, jit: bool = False
    ):
        assert (
            self._parallel_manager is not None
        ), "Parallel manager is not initialized!"

        fn_name = str(id(self)) + fn_name
        return self._parallel_manager.register_callable(fn, fn_name, jit)

    def _run_callable(self, *primals, fn_name: str):
        assert (
            self._parallel_manager is not None
        ), "Parallel manager is not initialized!"

        fn_name = str(id(self)) + fn_name
        return self._parallel_manager.run_callable(*primals, fn_name=fn_name)

    def _create_parallel(self, device_mesh: tuple[int, ...]):
        self._parallel_manager = JaxParallel(math.prod(device_mesh), self._device)

    def array(
        self,
        input: Any,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        result = self._conversion_fn_wrapper(jax.numpy.array)(
            input, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )
        return result

    def zeros(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        result = self._creation_fn_wrapper(jax.numpy.zeros)(
            _shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )
        return result

    def ones(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        result = self._creation_fn_wrapper(jax.numpy.ones)(
            _shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )
        return result

    def ones_like(
        self,
        input: jax.Array,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        result = self._creation_fn_wrapper(jax.numpy.ones_like)(
            input, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )
        return result

    def zeros_like(
        self,
        input: jax.Array,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> jax.Array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        result = self._creation_fn_wrapper(jax.numpy.zeros_like)(
            input, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )
        return result

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> jax.Array:
        if prng_key is None:
            prng_key = self.prng_key
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        result = self._creation_fn_wrapper(jax.random.normal)(
            prng_key, _shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )
        return result

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> jax.Array:
        if prng_key is None:
            prng_key = self.prng_key
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        result = self._creation_fn_wrapper(jax.random.uniform)(
            prng_key, _shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )
        return result

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> jax.Array:
        if prng_key is None:
            prng_key = self.prng_key
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        result = self._creation_fn_wrapper(jax.random.randint)(
            prng_key,
            _shape,
            low,
            high,
            dtype=utils.dtype_map[_dtype],
            device_mesh=device_mesh,
        )
        return result

    def rand_uniform(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> jax.Array:
        if prng_key is None:
            prng_key = self.prng_key
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(jax.random.uniform)(
            prng_key,
            _shape,
            dtype=utils.dtype_map[_dtype],
            minval=low,
            maxval=high,
            device_mesh=device_mesh,
        )

    def _arange(
        self,
        *args,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        **kwargs,
    ) -> jax.Array:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(jax.numpy.arange)(
            *args, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def to_numpy(self, arr: jax.Array) -> np.ndarray:
        return np.array(arr)

    def flatten(
        self, input: jax.Array, start_dim: int = 0, end_dim: int = -1
    ) -> jax.Array:
        return ops.flatten(input, start_dim=start_dim, end_dim=end_dim)

    def abs(self, array: jax.Array) -> jax.Array:
        return jax.numpy.abs(array)

    def sign(self, input: jax.Array) -> jax.Array:
        return jax.numpy.sign(input)

    def sin(self, array: jax.Array) -> jax.Array:
        return jax.numpy.sin(array)

    def cos(self, array: jax.Array) -> jax.Array:
        return jax.numpy.cos(array)

    def tanh(self, array: jax.Array) -> jax.Array:
        return jax.nn.tanh(array)

    def relu(self, array: jax.Array) -> jax.Array:
        return jax.nn.relu(array)

    def leaky_relu(self, array: jax.Array, slope: float | jax.Array) -> jax.Array:
        return jax.nn.leaky_relu(array, slope)

    def sigmoid(self, array: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(array)

    def softplus(self, array: jax.Array) -> jax.Array:
        return jax.nn.softplus(array)

    def softmax(self, input: jax.Array, dim: int = -1) -> jax.Array:
        # TODO: dim can be Sequence[int] as well. Should work
        # for all backends.
        return ops.softmax(input, axis=dim)

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
        self, inputs: tuple[jax.Array, ...] | list[jax.Array], dim: int = 0
    ) -> jax.Array:
        return ops.concat(*inputs, axis=dim)

    def pad(self, input: jax.Array, pad_width: PadWidthType) -> jax.Array:
        return jax.numpy.pad(input, pad_width)

    def all(self, input: jax.Array) -> jax.Array:
        return jax.numpy.all(input)

    def any(self, input: jax.Array) -> jax.Array:
        return jax.numpy.any(input)

    def atleast_1d(self, *inputs) -> jax.Array | list[jax.Array]:
        return jax.numpy.atleast_1d(*inputs)

    def atleast_2d(self, *inputs) -> jax.Array | list[jax.Array]:
        return jax.numpy.atleast_2d(*inputs)

    def transpose(
        self, data: jax.Array, axes: tuple[int, ...] | list[int] | None = None
    ) -> jax.Array:
        return data.transpose(axes)

    def unique(
        self, input, **kwargs
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
        return jax.numpy.unique(input, **kwargs)

    def where(self, cond: jax.Array, input1: jax.Array, input2: jax.Array) -> jax.Array:
        return ops.where(cond, input1, input2)

    def topk(self, input: jax.Array, k: int) -> jax.Array:
        return jax.lax.top_k(input, k)[0]

    def multinomial(
        self, probs: jax.Array, num_samples: int, replacement: bool = False, **kwargs
    ) -> jax.Array:
        """
        Faster JAX implementation of multinomial sampling.

        Args:
            key: JAX PRNG key
            input: 1D or 2D array of probabilities
            num_samples: number of samples to draw
            replacement: whether to sample with replacement
        """
        input = jax.numpy.asarray(probs)
        if input.ndim == 1:
            input = input[None, :]
            squeeze_result = True
        else:
            squeeze_result = False

        # Normalize probabilities
        input = input / jax.numpy.sum(input, axis=-1, keepdims=True)

        if replacement:
            # Use categorical directly - much faster than choice
            samples = jax.random.categorical(
                self.prng_key,
                jax.numpy.log(jax.numpy.maximum(input, 1e-37)),  # avoid log(0)
                shape=(input.shape[0], num_samples),
            )
        else:
            # For without replacement, use Gumbel-max trick
            # This is much faster than using choice
            z = -jax.numpy.log(
                -jax.numpy.log(
                    jax.random.uniform(
                        self.prng_key,
                        shape=(input.shape[0], input.shape[1], num_samples),
                    )
                )
            )
            # Add log probabilities for Gumbel-max trick
            z = z + jax.numpy.log(jax.numpy.maximum(input, 1e-37))[..., None]
            # Get top k indices
            samples = jax.numpy.argsort(-z, axis=1)[:, :num_samples]

        # Update prng_key.
        self.prng_key, _ = jax.random.split(self.prng_key)

        if squeeze_result:
            samples = jax.numpy.squeeze(samples, axis=0)

        return samples

    def jit(self, *args, **kwargs):
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
    ) -> tuple[Sequence[jax.Array], Callable, Sequence[jax.Array]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, jax.Array]],
        primals: dict[str, jax.Array],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[dict[str, jax.Array], Callable, dict[str, jax.Array]]: ...

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
        dict[str, jax.Array] | list[jax.Array] | Callable,
        dict[str, jax.Array] | Sequence[jax.Array] | jax.Array,
    ]:
        _primals: list | dict | jax.Array = primals
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

    def vmap(
        self, fn: Callable[..., dict[str, jax.Array]]
    ) -> Callable[..., dict[str, jax.Array]]:
        return jax.vmap(fn)

    def jacrev(
        self, fn: Callable[..., dict[str, jax.Array]]
    ) -> Callable[..., dict[str, jax.Array]]:
        return jax.jacrev(fn)

    def jacfwd(
        self, fn: Callable[..., dict[str, jax.Array]]
    ) -> Callable[..., dict[str, jax.Array]]:
        return jax.jacfwd(fn)
