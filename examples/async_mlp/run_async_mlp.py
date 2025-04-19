import math
import warnings
from typing import Any

from async_mlp_model import create_mlp_async

import mithril as ml
from mithril import Backend
from mithril.models import PhysicalModel
from mithril.backends.with_autograd.torch_backend.parallel import TorchParallel

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

def run_model():
    # World definition.
    world_size = 4

    # Create backend.
    backend =  ml.TorchBackend()
    backend_parallel = create_parallel_backend(device_mesh=(1, world_size))
    rowwise_shard_mesh = ((0, 1),(0, world_size))
    colwise_shard_mesh = ((0,1),(1, world_size))

    # Model Configuration
    input_dim = 128
    mlp_async = create_mlp_async(input_dim)

    # Compile model.
    compiled_model = ml.compile(
        mlp_async,
        backend,
        data_keys={"input"},
        jit=False,
        use_short_namings=False
    )

    # Create weights
    fc_weight = backend.arange(0, (input_dim * input_dim * 4), dtype=backend.default_dtype).reshape([input_dim * 4, input_dim])
    fc_bias = backend.zeros([1, input_dim * 4], dtype=backend.default_dtype)
    proj_weight = backend.arange(0, (input_dim * input_dim * 4), dtype=backend.default_dtype).reshape([input_dim, input_dim * 4])
    proj_bias = backend.zeros([1, input_dim], dtype=backend.default_dtype)
    trainables = {
        "mlp_async_fc_weight": backend_parallel.array(fc_weight, device_mesh=rowwise_shard_mesh),
        "mlp_async_fc_bias": backend_parallel.array(fc_bias, device_mesh=colwise_shard_mesh),
        "mlp_async_proj_weight": backend_parallel.array(proj_weight, device_mesh=colwise_shard_mesh),
        "mlp_async_proj_bias": backend_parallel.array(proj_bias, device_mesh=colwise_shard_mesh),
    }

    # Create input
    input = backend.arange(0, input_dim, dtype=backend.default_dtype).reshape([1, input_dim])
    x = {"input": backend_parallel.array(input, device_mesh=colwise_shard_mesh)}

    # Print expected result.
    expected_result = (input @ fc_weight.T + fc_bias) @ proj_weight.T + proj_bias
    print("Expected result:", expected_result)
    
    # Run generation
    generate(compiled_model,trainables, x)

def generate(
    model: PhysicalModel[Any],
    weights: dict[str, ml.DataType],
    x: dict [str, ml.DataType],
):
    # Forward the model to get the output
    outputs = model.evaluate(weights, data=x)
    y = outputs["output"]
    print("Async MLP output:", y.full_tensor())

if __name__ == "__main__":
    run_model()