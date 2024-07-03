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
from collections.abc import Callable

import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from ...parallel import Parallel


class JaxParallel(Parallel[jax.numpy.ndarray]):
    def __init__(self, n_devices: int, device: str) -> None:
        available_devices = jax.device_count(device)
        assert available_devices == n_devices, (
            f"In Jax parallel all devices must be used. But '{n_devices}' is used,"
            f"'{available_devices}' available!"
        )
        super().__init__(n_devices)

    def run_callable(self, *primals, fn_name: str):
        return self.callables[fn_name](*primals)

    def parallelize(
        self, tensor: jax.Array, device_mesh: tuple[int, ...] | None = None
    ):
        # Jax reuqires math.prod(device_mesh) == n_devices. To replicate a dimension
        # call 'replicate' method of Positional Sharding Object. Therefore, we need to
        # transform user provided device mesh to the one that satisfies the condition,
        # and replicate the dimensions that are provided as 1 in the device mesh.

        replicate_dims = []

        _device_mesh = [1] * tensor.ndim if device_mesh is None else list(device_mesh)

        _device_mesh = _device_mesh + [1] * (tensor.ndim - len(_device_mesh))

        mesh_devices = math.prod(_device_mesh)
        unused_devices = self.n_devices // mesh_devices

        for idx, item in enumerate(_device_mesh):
            if item == 1:
                replicate_dims.append(idx)
                if unused_devices != 1:
                    _device_mesh[idx] = unused_devices
                    unused_devices = 1

        device_mesh = tuple(_device_mesh)
        devices = mesh_utils.create_device_mesh(device_mesh)
        sharding = PositionalSharding(devices)
        for replicate_dim in replicate_dims:
            sharding = sharding.replicate(replicate_dim)

        return jax.device_put(tensor, sharding)

    def register_callable(self, fn: Callable, fn_name: str, jit: bool):
        if jit:
            fn = jax.jit(fn)

        self.callables[fn_name] = fn

    def clean_up(self):
        self.callables = {}
