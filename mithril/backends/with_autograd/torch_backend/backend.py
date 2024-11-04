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
import torch
import torch.nn.functional as F  # noqa: N812
from torch._functorch.apis import grad as torch_grad
from torch._functorch.apis import grad_and_value as torch_grad_and_value
from torch._functorch.apis import vmap as torch_vmap
from torch._functorch.eager_transforms import jacfwd as torch_jacfwd
from torch._functorch.eager_transforms import jacrev as torch_jacrev
from torch._functorch.eager_transforms import vjp as torch_vjp
from torch.distributed._tensor import DTensor

from ....core import Dtype
from ...backend import PadWidthType, ParallelBackend
from ...utils import process_shape
from . import ops, utils
from .parallel import TorchParallel

__all__ = ["TorchBackend"]


class TorchBackend(ParallelBackend[torch.Tensor]):
    """TorchBackend: A backend implementation for the Mithril library.

    This backend provides integration with PyTorch.

    Parameters
    ----------
    device: str, optional
        The device on which to perform computations, default is "cpu".
    precision: int, optional
        The precision of the tensors, either 32 or 64, default is 32.
    """

    type = "torch"
    registered_primitives = {}
    primitive_fn_path = "mithril.backends.with_autograd.torch_backend.ops"

    def __init__(
        self, device: str = "cpu", precision: int = 32, device_mesh=None
    ) -> None:
        self._device = device
        self._precision = precision
        self._parallel_manager: TorchParallel | None = None

        utils.get_device(device)  # Check if device is valid

        super().__init__(device_mesh=device_mesh)
        if device_mesh is not None:
            self._create_parallel(device_mesh)

        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict

        torch.random.manual_seed(self.seed)

    @property
    def is_manualgrad(self):
        return False

    @property
    def inf(self):
        return torch.inf

    @property
    def DataType(self):  # noqa: N802
        return utils.ArrayType

    @property
    def device(self):
        return utils.get_device(self._device)

    def get_backend_array_type(self):
        return torch.Tensor

    @staticmethod
    def register_primitive(fn: Callable) -> None:
        TorchBackend.registered_primitives[fn.__name__] = fn

    @staticmethod
    def get_available_devices() -> list[str]:
        """Static method to get a list of available devices.

        Parameters
        ----------
        list[str]
            List of available devices.
        """

        return utils.get_available_devices()

    def set_seed(self, seed: int):
        self.seed = seed
        torch.random.manual_seed(seed)

    def to_device(self, data: torch.Tensor, device: str, asynchronous: bool = False):
        """Move data to the specified device.

        Parameters
        ----------
        data: torch.Tensor
            The data to be moved to the specified device.
        device: str
            The target device for the data.
        """
        return data.to(device)

    # def block_until_ready(self, data: ArrayType | None = None) -> None:
    #     getattr(torch, f"{self.device_type.lower()}").synchronize()

    def empty_cache(self) -> None:
        """Empty the cache on the device."""
        if self._device in ["MPS", "CUDA"]:
            getattr(torch, f"{self._device.lower()}").empty_cache()
        else:
            pass
            # print(f"Warning: empty_cache is not implemented for {self.device_type}")

    def _creation_fn_wrapper(self, fn: Callable) -> Callable:
        """
        Wrapper for PyTorch tensor creation functions.

        Parameters
        ----------
        fn: Callable
            The original tensor creation function.

        Returns
        -------
        Callable
            A wrapped function that creates tensors with specified dtype and device.

        Notes
        -----
        This wrapper ensures that tensors are created with the correct dtype
        and on the specified device.
        """

        array_creation_fn = partial(
            utils.creation_fn_wrapper_inner,
            fn=fn,
            device=self._device,
            precision=self.precision,
        )
        array_creation_fn = partial(self._parallelize, fn=array_creation_fn)

        return array_creation_fn

    def _conversion_fn_wrapper(self, fn: Callable) -> Callable:
        """
        Wrapper for PyTorch tensor conversion functions.

        Parameters
        ----------
        fn: Callable
            The original tensor conversion function.

        Returns
        -------
        Callable
            A wrapped function that converts tensors with specified dtype and device.

        Notes
        -----
        Wrapper handles the conversion of tensors between different dtypes and devices.
        """

        array_conversion_fn = partial(
            utils.conversion_fn_wrapper_inner,
            fn=fn,
            device=self._device,
            precision=self.precision,
        )
        array_conversion_fn = partial(self._parallelize, fn=array_conversion_fn)

        return array_conversion_fn

    def _parallelize(
        self, *args, fn: Callable, device_mesh, **kwargs
    ) -> DTensor | torch.Tensor:
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
            Returns tensor parallelized across the specified device mesh.
        """
        tensor: torch.Tensor = fn(*args, **kwargs)
        if self._parallel_manager is None:
            # TODO: raise device_mesh should be None
            return tensor

        return self._parallel_manager.parallelize(
            tensor, self.base_device_mesh, device_mesh
        )

    def _register_callable(
        self, fn: Callable | partial, fn_name: str, jit: bool = False
    ):
        """
        Register a callable function with the backend.

        Parameters
        ----------
        fn: Callable
            The function to be registered.
        """
        assert (
            self._parallel_manager is not None
        ), "Parallel manager is not initialized!"

        fn_name = str(id(self)) + fn_name
        self._parallel_manager.register_callable(
            fn, fn_name, self.base_device_mesh, jit
        )

    def _create_parallel(self, device_mesh: tuple[int, ...]):
        assert isinstance(device_mesh, tuple), "Device mesh must be tuple or None!"
        assert isinstance(
            self._raw_device_mesh, tuple
        ), "Device mesh must be tuple or None!"

        self._parallel_manager = TorchParallel(
            self.n_devices, device=self._device.split(":")[0]
        )
        self.base_device_mesh = self._parallel_manager._init_device_mesh(
            self._raw_device_mesh
        )

    def _run_callable(self, *primals, fn_name: str):
        assert (
            self._parallel_manager is not None
        ), "Parallel manager is not initialized!"

        fn_name = str(id(self)) + fn_name
        return self._parallel_manager.run_callable(*primals, fn_name=fn_name)

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        # _parallel_manager is not picklable, not going to write it into pickle file
        # We can recreate it using the device mesh
        if "_parallel_manager" in state:
            del state["_parallel_manager"]
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        # Recreate the parallel manager
        if self._raw_device_mesh is not None:
            self._create_parallel(self._raw_device_mesh)
        else:
            self._parallel_manager = None

    def array(
        self,
        input: Any,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._conversion_fn_wrapper(torch.tensor)(
            input, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def zeros(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(torch.zeros)(
            _shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def ones(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(torch.ones)(
            _shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def ones_like(
        self,
        input: torch.Tensor,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(torch.ones_like)(
            input, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def zeros_like(
        self,
        input: torch.Tensor,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(torch.zeros_like)(
            input, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(torch.randn)(
            size=_shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(torch.rand)(
            size=_shape, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        _shape = process_shape(shape)
        return self._creation_fn_wrapper(torch.randint)(
            low,
            high,
            size=_shape,
            dtype=utils.dtype_map[_dtype],
            device_mesh=device_mesh,
        )

    def rand_uniform(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        prng_key: Any = None,
    ) -> torch.Tensor:
        return (low - high) * self.rand(
            *shape, dtype=dtype, device_mesh=device_mesh
        ) + high

    def _arange(
        self,
        *args,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        _dtype: str | None = None
        if isinstance(dtype, Dtype):
            _dtype = dtype.name
        return self._creation_fn_wrapper(torch.arange)(
            *args, dtype=utils.dtype_map[_dtype], device_mesh=device_mesh
        )

    def to_numpy(self, arr: torch.Tensor) -> np.ndarray:
        return arr.detach().cpu().numpy()

    def flatten(
        self, input: torch.Tensor, start_dim: int = 0, end_dim: int = -1
    ) -> torch.Tensor:
        return torch.flatten(input, start_dim=start_dim, end_dim=end_dim)

    def abs(self, input: torch.Tensor) -> torch.Tensor:
        return torch.abs(input)

    def sign(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sign(input)

    def sin(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)

    def cos(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cos(input)

    def tanh(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input)

    def relu(self, input: torch.Tensor) -> torch.Tensor:
        return torch.relu(input)

    def leaky_relu(
        self, input: torch.Tensor, slope: float | torch.Tensor
    ) -> torch.Tensor:
        return F.leaky_relu(input, negative_slope=slope)  # type: ignore

    def sigmoid(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(input)

    def softplus(self, input: torch.Tensor) -> torch.Tensor:
        return F.softplus(input)

    def softmax(self, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return ops.softmax(input, axis=dim)

    def log(self, input: torch.Tensor) -> torch.Tensor:
        return torch.log(input)

    def isnan(self, input: torch.Tensor) -> torch.Tensor:
        return torch.isnan(input)

    def stop_gradient(self, input: torch.Tensor) -> torch.Tensor:
        return input.detach()

    def squeeze(self, input: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(input)

    def reshape(self, input: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.reshape(input, shape)

    def sort(
        self, input: torch.Tensor, axis: int = -1, descending: bool = False
    ) -> torch.Tensor:
        return torch.sort(input, dim=axis, descending=descending).values

    def expand_dims(self, input: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.unsqueeze(input, axis)

    def stack(self, inputs: list[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.stack(inputs, dim=axis)

    def cat(
        self, inputs: tuple[torch.Tensor, ...] | list[torch.Tensor], axis: int = 0
    ) -> torch.Tensor:
        return ops.concat(*inputs, axis=axis)

    def pad(self, input: torch.Tensor, pad_width: PadWidthType) -> torch.Tensor:
        assert isinstance(pad_width, tuple)
        assert isinstance(pad_width[0], int)
        return F.pad(input, pad_width)

    def all(self, input: torch.Tensor) -> torch.Tensor:
        return torch.all(input)

    def any(self, input: torch.Tensor) -> torch.Tensor:
        return torch.any(input)

    def atleast_1d(self, *inputs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return torch.atleast_1d(inputs)

    def atleast_2d(self, *inputs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return torch.atleast_2d(inputs)

    def transpose(
        self, input: torch.Tensor, axes: tuple[int, ...] | list[int] | None
    ) -> torch.Tensor:
        return ops.transpose(input, axes)

    def unique(
        self, input, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return torch.unique(input, **kwargs)

    def where(
        self, cond: torch.Tensor, input1: torch.Tensor, input2: torch.Tensor
    ) -> torch.Tensor:
        return ops.where(cond, input1, input2)

    def topk(self, input: torch.Tensor, k: int) -> torch.Tensor:
        return torch.topk(input, k)[0]  # TODO: Returns different tuple type???

    def multinomial(
        self,
        probs: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return torch.multinomial(probs, num_samples, replacement, **kwargs)

    def jit(self, *args, **kwargs):
        backend = "inductor"
        if "mps" in self._device:
            backend = "aot_eager"
        return torch.compile(*args, backend=backend, **kwargs)

    def grad(self, fn: Callable) -> Callable:
        return torch_grad(fn)

    def value_and_grad(
        self, fn: Callable
    ) -> Callable[..., tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        return torch_grad_and_value(fn)

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]],
        primals: list[torch.Tensor],
        *,
        cotangents: tuple[torch.Tensor, ...],
        has_aux: bool = True,
    ) -> tuple[Sequence[torch.Tensor], list[torch.Tensor], Sequence[torch.Tensor]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
        primals: dict[str, torch.Tensor],
        *,
        cotangents: dict[str, torch.Tensor],
        has_aux: bool = True,
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[torch.Tensor]],
        primals: list[torch.Tensor],
        *,
        cotangents: tuple[torch.Tensor, ...],
        has_aux: bool = False,
    ) -> tuple[Sequence[torch.Tensor], list[torch.Tensor], Sequence[torch.Tensor]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, torch.Tensor]],
        primals: dict[str, torch.Tensor],
        *,
        cotangents: dict[str, torch.Tensor],
        has_aux: bool = False,
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., Sequence[torch.Tensor]],
        primals: list[torch.Tensor],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[Sequence[torch.Tensor], Callable, Sequence[torch.Tensor]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, torch.Tensor]],
        primals: dict[str, torch.Tensor],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[dict[str, torch.Tensor], Callable, dict[str, torch.Tensor]]: ...

    def vjp(
        self,
        fn: Callable[
            ...,
            dict[str, torch.Tensor]
            | Sequence[torch.Tensor]
            | tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]
            | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        ],
        primals: dict[str, torch.Tensor] | list[torch.Tensor],
        *,
        cotangents: dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        has_aux: bool = False,
    ) -> tuple[
        dict[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
        dict[str, torch.Tensor] | list[torch.Tensor] | Callable,
        dict[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    ]:
        _primals: list | dict | torch.Tensor = primals
        if isinstance(primals, dict | torch.Tensor):
            _primals = [primals]
        output, vjp, *aux = torch_vjp(fn, *_primals, has_aux=has_aux)
        if has_aux:
            (aux,) = aux
        else:
            aux = {} if isinstance(cotangents, dict) else []
        if cotangents is not None:
            vjp = vjp(cotangents)
            if isinstance(cotangents, dict):
                # Torch vjp returns tuple[dict] for dict type returns.
                # So we should unpack vjp result.
                (vjp,) = vjp
        return output, vjp, aux

    def vmap(self, fn: Callable[..., dict[str, torch.Tensor]]) -> Callable:
        return torch_vmap(fn)

    def jacrev(self, fn: Callable[..., dict[str, torch.Tensor]]) -> Callable:
        return torch_jacrev(fn)

    def jacfwd(self, fn: Callable[..., dict[str, torch.Tensor]]) -> Callable:
        return torch_jacfwd(fn)
