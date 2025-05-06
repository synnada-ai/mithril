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
import warnings
from typing import Any

import mithril as ml
from async_mlp_model import create_mlp_async
from mithril.backends.with_autograd.torch_backend.parallel import TorchParallel
from mithril.models import PhysicalModel

warnings.filterwarnings("ignore")

current_device_mesh: tuple[int, ...] | None = None


def create_parallel_backend(device_mesh: tuple[int, ...]):
    global current_device_mesh
    if current_device_mesh is not None and math.prod(device_mesh) != math.prod(
        current_device_mesh
    ):
        if TorchParallel._instance is not None:
            TorchParallel._instance.clean_up()
        current_device_mesh = None

    backend = ml.TorchBackend(device_mesh=device_mesh)
    current_device_mesh = device_mesh
    return backend


def run_model(world_size: int = 4, input_dim: int = 1024) -> Any:
    # Create backend.
    backend = ml.TorchBackend()
    backend_parallel = create_parallel_backend(device_mesh=(1, world_size))
    rowwise_shard_mesh = ((0, 1), (0, world_size))
    colwise_shard_mesh = ((0, 1), (1, world_size))

    # Model initialization
    mlp_async = create_mlp_async(input_dim)

    # Compile model
    compiled_model = ml.compile(
        mlp_async, backend, data_keys={"input"}, jit=False, use_short_namings=False
    )

    # Create weights
    fc_weight = backend.arange(
        0, (input_dim * input_dim * 4), dtype=backend.default_dtype
    ).reshape([input_dim * 4, input_dim])
    fc_bias = backend.zeros([1, input_dim * 4], dtype=backend.default_dtype)
    proj_weight = backend.arange(
        0, (input_dim * input_dim * 4), dtype=backend.default_dtype
    ).reshape([input_dim, input_dim * 4])
    proj_bias = backend.zeros([1, input_dim], dtype=backend.default_dtype)
    trainables = {
        "mlp_async_fc_weight": backend_parallel.array(
            fc_weight, device_mesh=rowwise_shard_mesh
        ),
        "mlp_async_fc_bias": backend_parallel.array(
            fc_bias, device_mesh=colwise_shard_mesh
        ),
        "mlp_async_proj_weight": backend_parallel.array(
            proj_weight, device_mesh=colwise_shard_mesh
        ),
        "mlp_async_proj_bias": backend_parallel.array(
            proj_bias, device_mesh=colwise_shard_mesh
        ),
    }

    # Create input
    input = backend.arange(
        0, input_dim * input_dim, dtype=backend.default_dtype
    ).reshape([input_dim, input_dim])
    x = {"input": backend_parallel.array(input, device_mesh=colwise_shard_mesh)}

    # Run generation
    output = generate(compiled_model, trainables, x)

    return output


def generate(
    model: PhysicalModel[Any],
    weights: dict[str, ml.DataType],
    x: dict[str, ml.DataType],
) -> Any:
    # Forward the model to get the output
    outputs = model.evaluate(weights, data=x)
    return outputs["output"]


if __name__ == "__main__":
    output = run_model().full_tensor()
    print(f"Output: {output}")
