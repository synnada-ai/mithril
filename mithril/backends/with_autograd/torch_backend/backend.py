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
from typing import Any, overload

import torch
import torch.nn.functional as F  # noqa: N812
from torch._functorch.apis import grad as torch_grad
from torch._functorch.apis import grad_and_value as torch_grad_and_value
from torch._functorch.apis import vmap as torch_vmap
from torch._functorch.eager_transforms import jacfwd as torch_jacfwd
from torch._functorch.eager_transforms import jacrev as torch_jacrev
from torch._functorch.eager_transforms import vjp as torch_vjp

from ....core import Dtype
from ...backend import PadWidthType, ParallelBackend
from ...utils import DtypeSubTypes, StaticScalar, process_shape
from . import ops, utils
from .parallel import TorchParallel
from .utils import CODEGEN_CONFIG

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

    backend_type = "torch"
    registered_primitives = {}
    primitive_fn_path = "mithril.backends.with_autograd.torch_backend.ops"

    def __init__(
        self,
        device: str = "cpu",
        dtype: Dtype = Dtype.float32,
        device_mesh: tuple[int, ...] | None = None,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._parallel_manager: TorchParallel | None = None

        utils.get_device(device)  # Check if device is valid

        super().__init__(dtype=dtype, device_mesh=device_mesh)
        if device_mesh is not None:
            self._create_parallel(device_mesh)

        self.array_creation_funcs = ops.array_creation_funcs
        self.primitive_function_dict = ops.primitive_func_dict

        self._generator = torch.Generator(device=self.device).manual_seed(0)

        for key, value in utils.dtype_map.items():
            setattr(self, key, value)

    @property
    def is_manualgrad(self) -> bool:
        return False

    @property
    def inf(self) -> float:
        return torch.inf

    @property
    def nan(self) -> float:
        return torch.nan

    @property
    def DataType(self) -> type[torch.Tensor]:  # noqa: N802
        return utils.ArrayType

    @property
    def codegen_config(self) -> dict[str, bool]:
        return CODEGEN_CONFIG

    @property
    def device(self) -> torch.device:
        return utils.get_device(self._device)

    def get_backend_array_type(self) -> type[torch.Tensor]:
        return torch.Tensor

    def get_device(self) -> Any:
        return self._device

    @staticmethod
    def register_primitive(fn: Callable[..., Any]) -> None:
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

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self._generator.manual_seed(seed)

    def to_device(
        self, data: torch.Tensor, device: str, asynchronous: bool = False
    ) -> torch.Tensor:
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

    def register_callable(
        self, fn: Callable[..., torch.Tensor], fn_name: str, jit: bool = False
    ) -> None:
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

    def _create_parallel(self, device_mesh: tuple[int, ...]) -> None:
        assert isinstance(device_mesh, tuple), "Device mesh must be tuple or None!"
        assert isinstance(
            self._raw_device_mesh, tuple
        ), "Device mesh must be tuple or None!"

        self._parallel_manager = TorchParallel(
            self.n_devices, device=self._device.split(":")[0]
        )
        self.base_device_mesh = self._parallel_manager.init_device_mesh(
            self._raw_device_mesh
        )

    def _run_callable(self, *primals: Any, fn_name: str) -> Any:
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

    def __setstate__(self, state: dict[Any, Any]) -> None:
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
        _dtype = utils.determine_dtype(input, dtype, self._dtype, self.precision)

        array = torch.tensor(input, dtype=utils.dtype_map[_dtype], device=self._device)
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )

        return array

    def zeros(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        array = torch.zeros(_shape, dtype=_dtype, device=self._device)
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )

        return array

    def ones(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        array = torch.ones(_shape, dtype=_dtype, device=self._device)
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )
        return array

    def ones_like(
        self,
        input: torch.Tensor,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype = self._process_dtype(dtype) if dtype is not None else None

        array = torch.ones_like(input, dtype=_dtype, device=self._device)
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )
        return array

    def zeros_like(
        self,
        input: torch.Tensor,
        *,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype = self._process_dtype(dtype) if dtype is not None else None

        array = torch.zeros_like(input, dtype=_dtype, device=self._device)
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )
        return array

    def randn(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> torch.Tensor:
        generator = self._get_generator(key)

        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        # TODO: PRNG key is not used
        array = torch.randn(
            _shape, dtype=_dtype, device=self._device, generator=generator
        )
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )
        return array

    def rand(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> torch.Tensor:
        generator = self._get_generator(key)
        _dtype = self._process_dtype(dtype)
        _shape = process_shape(shape)

        array = torch.rand(
            _shape, dtype=_dtype, device=self._device, generator=generator
        )
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )
        return array

    def randint(
        self,
        low: int,
        high: int,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> torch.Tensor:
        generator = self._get_generator(key)
        _dtype = self._process_dtype(dtype, "int")
        _shape = process_shape(shape)

        array = torch.randint(
            low, high, _shape, dtype=_dtype, device=self._device, generator=generator
        )
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )
        return array

    def rand_uniform(
        self,
        low: int | float | bool | torch.Tensor,
        high: int | float | bool | torch.Tensor,
        *shape: int | tuple[int, ...] | list[int],
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
        key: int | None = None,
    ) -> torch.Tensor:
        return (low - high) * self.rand(
            *shape, dtype=dtype, device_mesh=device_mesh, key=key
        ) + high

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        default_type = (
            self._get_default_subtype()
            if any(isinstance(x, float) for x in (start, stop, step))
            else "int"
        )
        _dtype = self._process_dtype(dtype, default_type)

        array = torch.arange(start, stop, step, dtype=_dtype, device=self._device)
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )

        return array

    def linspace(
        self,
        start: int | float | bool | torch.Tensor,
        stop: int | float | bool | torch.Tensor,
        steps: int,
        dtype: Dtype | None = None,
        device_mesh: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        _dtype = self._process_dtype(dtype)

        array = torch.linspace(start, stop, steps, dtype=_dtype, device=self._device)
        if self._parallel_manager is not None:
            array = self._parallel_manager.parallelize(
                array, self.base_device_mesh, device_mesh
            )
        return array

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

    def argmax(
        self, input: torch.Tensor, axis: int | None = None, keepdim: bool = False
    ) -> torch.Tensor:
        return torch.argmax(input, dim=axis, keepdim=keepdim)

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
        _pad: list[int] = []
        assert isinstance(pad_width, tuple | list | int)

        # Standardize torch's pad with other backends
        # torch's pad takes Sequence[int] or Tuple[int, ...]
        # other backends takes PadWidthType

        # If pad comes with PadWidthType, flat and convert it to
        # torch's pad format.
        if isinstance(pad_width, int):
            # if int is given, pad all dimensions
            for _ in range(len(input.shape)):
                _pad.extend([pad_width, pad_width])

        elif isinstance(pad_width[0], tuple):
            # if pad is given with tuple[tuple[int, int], ...]
            # pad all befores and afters in each dims
            for pad_dim in pad_width[::-1]:
                assert isinstance(pad_dim, tuple)
                _pad.extend(pad_dim)
        else:
            # if pad is given with tuple[int, int]
            # pad all befores and afters in all dims
            for _ in range(len(input.shape)):
                _pad.extend(pad_width)  # type: ignore

        return F.pad(input, _pad)

    def all(self, input: torch.Tensor) -> torch.Tensor:
        return torch.all(input)

    def any(self, input: torch.Tensor) -> torch.Tensor:
        return torch.any(input)

    def atleast_1d(
        self, inputs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(inputs, tuple):
            return torch.atleast_1d(*inputs)  # type: ignore
        else:
            return torch.atleast_1d(inputs)  # type: ignore

    def atleast_2d(
        self, inputs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(inputs, tuple):
            return torch.atleast_2d(*inputs)  # type: ignore
        else:
            return torch.atleast_2d(inputs)  # type: ignore

    def transpose(
        self, input: torch.Tensor, axes: tuple[int, ...] | list[int] | None = None
    ) -> torch.Tensor:
        return ops.transpose(input, axes)

    def unique(
        self, input: torch.Tensor, **kwargs: Any
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
        key: int | None = None,
    ) -> torch.Tensor:
        return torch.multinomial(
            probs, num_samples, replacement, generator=self._get_generator(key)
        )

    def clip(
        self,
        input: torch.Tensor,
        min: torch.Tensor | StaticScalar,
        max: torch.Tensor | StaticScalar,
    ) -> torch.Tensor:
        return torch.clamp(input, min, max)  # type: ignore [arg-type]

    def jit(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        backend = "inductor"
        if "mps" in self._device:
            backend = "aot_eager"
        return torch.compile(*args, backend=backend, **kwargs)

    def grad(
        self, fn: Callable[..., dict[str, torch.Tensor]]
    ) -> Callable[..., dict[str, torch.Tensor]]:
        return torch_grad(fn)

    def value_and_grad(
        self, fn: Callable[..., dict[str, torch.Tensor]]
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
    ) -> tuple[Sequence[torch.Tensor], Callable[..., Any], Sequence[torch.Tensor]]: ...

    @overload
    def vjp(
        self,
        fn: Callable[..., dict[str, torch.Tensor]],
        primals: dict[str, torch.Tensor],
        *,
        cotangents: None,
        has_aux: bool = False,
    ) -> tuple[
        dict[str, torch.Tensor], Callable[..., Any], dict[str, torch.Tensor]
    ]: ...

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
        dict[str, torch.Tensor] | list[torch.Tensor] | Callable[..., Any],
        dict[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    ]:
        _primals: list[torch.Tensor] | dict[str, torch.Tensor] | torch.Tensor = primals
        if isinstance(primals, dict | torch.Tensor):
            _primals = [primals]  # type: ignore
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

    def vmap(  # type: ignore  # mypy bug
        self, fn: Callable[..., dict[str, torch.Tensor]]
    ) -> Callable[..., dict[str, torch.Tensor]]:
        return torch_vmap(fn)

    def jacrev(self, fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        return torch_jacrev(fn)

    def jacfwd(self, fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        return torch_jacfwd(fn)

    def _get_generator(self, key: int | None) -> torch.Generator:
        if key is None:
            return self._generator
        else:
            return torch.Generator(device=self.device).manual_seed(key)

    def _process_dtype(
        self,
        dtype: Dtype | None = None,
        default_type: str | None = None,
    ) -> torch.dtype:
        if isinstance(dtype, Dtype):
            return utils.dtype_map[dtype.name]
        elif dtype is None:
            if default_type is None:
                default_type = self._get_default_subtype()
            return utils.dtype_map[default_type + str(self.precision)]
        else:
            raise ValueError(f"Invalid dtype {dtype}")

    def _get_default_subtype(self) -> str:
        return DtypeSubTypes[self._dtype.name].value
