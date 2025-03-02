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

import jax
import jax.random as random
from mithril.backends.with_autograd.jax_backend.utils import dtype_map
from mithril.cores.python.jax.utils import get_device

from mithril.framework.common import Tensor, NOT_GIVEN, TBD, ToBeDetermined
from mithril.framework.logical.base import BaseKey
from mithril.models import PrimitiveModel, Connection, ExtendInfo, ConnectionType, Model
import mithril as ml

def test_random_state_model():
    def update_random_key(key: jax.random.PRNGKey) -> jax.random.PRNGKey:
        # Generate a new PRNG key by splitting the given key
        new_prng_key, _ = random.split(key)
        return new_prng_key

    def my_randn(
        shape: tuple[int, ...],
        key: jax.random.PRNGKey,
        *,
        dtype: str | None = None,
    ) -> jax.Array:
        with jax.default_device(get_device('cpu')):
            return jax.random.normal(key, shape, dtype="float32")
    

    ml.JaxBackend.register_primitive(update_random_key)
    ml.JaxBackend.register_primitive(my_randn)
            
    class MyRandn(PrimitiveModel):
        shape: Connection
        key: Connection
        dtype: Connection
        output: Connection

        def __init__(
            self,
            shape: tuple[int, ...] | ToBeDetermined = TBD,
            key: int | ToBeDetermined = TBD,
            dtype: ml.types.Dtype | None = None,
            *,
            name: str | None = None,
        ) -> None:
            super().__init__(
                formula_key="my_randn",
                name=name,
                output=BaseKey(shape=[("output", ...)], type=Tensor),
                shape=BaseKey(type=tuple[int, ...], value=shape),
                key=BaseKey(type=int, value=key),
                dtype=BaseKey(type=ml.types.Dtype | None, value=dtype),
            )

    class UpdateRandomKey(PrimitiveModel):
        key: Connection
        output: Connection

        def __init__(
            self,
            key: int | ToBeDetermined = TBD,
            *,
            name: str | None = None,
        ) -> None:
            super().__init__(
                formula_key="update_random_key",
                name=name,
                output=BaseKey(type=int, value=key),
                key=BaseKey(type=int, value=key),
            )

    model = Model()
    model |= UpdateRandomKey()(key="key", output="new_key")
    model |= MyRandn()(shape = "shape", key="new_key")
    # model.set_as_state_output("new_key", "key")

    pm = ml.compile(model, ml.JaxBackend(), jit=False)
    prng = random.PRNGKey(42)
    res = pm.evaluate(data={"shape": (1, 1), "key": prng})


# import torch

# # Create a generator with a specific seed
# seed = 42
# generator = torch.Generator().manual_seed(seed)

# # Split the generator into two new generators
# def split_generator(generator, num_splits):
#     generators = []
#     for _ in range(num_splits):
#         # Create a new generator with a new seed derived from the original generator
#         new_seed = torch.randint(0, 2**32, (1,), generator=generator).item()
#         generators.append(torch.Generator().manual_seed(new_seed))
#     return generators

# # Split the generator into two new generators
# generator1, generator2 = split_generator(generator, 2)

# # Use the generators to generate random numbers
# random_tensor1 = torch.randn(3, 3, generator=generator1)
# random_tensor2 = torch.randn(3, 3, generator=generator2)

# print(random_tensor1)
# print(random_tensor2)



# import numpy as np

# # Create a SeedSequence from a seed
# seed = 42
# ss = np.random.SeedSequence(seed)

# # Split the SeedSequence into multiple independent streams
# child_seeds = ss.spawn(2)  # Spawn 2 independent seeds

# # Create independent generators for each seed
# rng1 = np.random.default_rng(child_seeds[0])
# rng2 = np.random.default_rng(child_seeds[1])

# # Generate random numbers
# random_array1 = rng1.random((3, 3))
# random_array2 = rng2.random((3, 3))

# print("Random array 1:\n", random_array1)
# print("Random array 2:\n", random_array2)



# import mlx.core as mx
# import mlx.core.random as random

# # Create a random key
# key = random.key(42)

# # Split the key into multiple keys
# key1, key2 = random.split(key)

# # Generate random numbers using the split keys
# random_array1 = random.normal(shape=(3, 3), key=key1)
# random_array2 = random.normal(shape=(3, 3), key=key2)

# print("Random array 1:\n", random_array1)
# print("Random array 2:\n", random_array2)


import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

# Loop-based comparison test
def test_batchnorm_vs_pytorch():
    class BatchNorm:
        def __init__(self, num_features, momentum=0.1, eps=1e-5):
            self.momentum = momentum
            self.eps = eps
            self.gamma = jnp.ones(num_features)  # Scale parameter
            self.beta = jnp.zeros(num_features)  # Shift parameter
            self.running_mean = jnp.zeros(num_features)  # Running mean
            self.running_var = jnp.ones(num_features)   # Running variance

        def __call__(self, x, training=True):
            if training:
                # Compute batch statistics
                batch_mean = jnp.mean(x, axis=0)
                batch_var = jnp.sum((x - batch_mean) ** 2, axis=0) / (x.shape[0] - 1)
                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean +  self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                
                # Normalize using batch statistics
                x_norm = (x - batch_mean) / jnp.sqrt(batch_var + self.eps)
            else:
                # Normalize using running statistics
                x_norm = (x - self.running_mean) / jnp.sqrt(self.running_var + self.eps)
            
            # Scale and shift
            return self.gamma * x_norm + self.beta

    # PyTorch BatchNorm for comparison
    class PyTorchBatchNorm:
        def __init__(self, num_features, momentum=0.1, eps=1e-5):
            self.batchnorm = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps, affine=False)
        
        def __call__(self, x, training=True):
            if training:
                self.batchnorm.train()
            else:
                self.batchnorm.eval()
            return self.batchnorm(x)

    # Generate random data for testing
    def generate_data(batch_size, num_features, num_batches):
        rng = np.random.RandomState(42)
        data = [rng.randn(batch_size, num_features).astype(np.float32) for _ in range(num_batches)]
        return data
    
    # Hyperparameters
    num_features = 2
    batch_size = 4
    num_batches = 10
    momentum = 0.1
    eps = 1e-5

    # Initialize BatchNorm layers
    batchnorm_jax = BatchNorm(num_features, momentum=momentum, eps=eps)
    batchnorm_torch = PyTorchBatchNorm(num_features, momentum=momentum, eps=eps)

    # Generate random data
    data = generate_data(batch_size, num_features, num_batches)

    # Training loop
    for i, x_np in enumerate(data):
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        # JAX BatchNorm
        output_jax = batchnorm_jax(x_jax, training=True)
        
        # PyTorch BatchNorm
        output_torch = batchnorm_torch(x_torch, training=True)

        # Compare outputs
        assert jnp.allclose(output_jax, output_torch.detach().numpy(), atol=1e-5), f"Output mismatch at batch {i}"

        # Compare running statistics
        assert jnp.allclose(batchnorm_jax.running_mean, batchnorm_torch.batchnorm.running_mean.numpy(), atol=1e-5), f"Running mean mismatch at batch {i}"
        assert jnp.allclose(batchnorm_jax.running_var, batchnorm_torch.batchnorm.running_var.numpy(), atol=1e-5), f"Running var mismatch at batch {i}"

    print("All tests passed! Running statistics and outputs match between JAX and PyTorch.")

# def test_running_mean():
#     import mithril as ml
#     from mithril.models import Buffer, Square, Mean
#     momentum = ml.IOKey("momentum")
#     running_mean = ml.IOKey("running_mean")
#     batch_mean = ml.IOKey("batch_mean")

#     model = Model()
#     model |= Mean()(input="input", output=batch_mean)
#     model |= Buffer()((1 - momentum) * running_mean + momentum * batch_mean, output="new_mean")
#     model.bind_state_input("running_mean", "new_mean", 0)

#     main_model = Model()
#     main_model |= model(input="input", momentum=0.1, new_mean="mean")
#     main_model |= Square()(input="mean", output=ml.IOKey("output"))

#     pm = ml.compile(main_model, ml.JaxBackend(), jit=False, use_short_namings=False, inference=True, file_path="generated.py")
#     res = pm.evaluate(data={"input": jax.numpy.ones((1, 1)), "running_mean": jax.numpy.zeros((1, 1))})
#     ...
import mithril as ml
from mithril.models import Buffer, Add

def test_running_mean():
    # Manually expose state inputs & outputs and manually 
    # update running_output in the evaluation loop.
    model = Model()
    model |= Add()("input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="output")

    main_model = Model()
    main_model |= model(input="input", running_input="running_input", output="running_output")
    main_model |= Buffer()("running_output", output="output")
    main_model.expose_keys("running_output", "output")
    # TODO: use_short_namings -> short_names
    pm = ml.compile(main_model, ml.JaxBackend(), jit=False, use_short_namings=False, inference=True)
    manual_state = jax.numpy.ones((1, 1))
    for _ in range(10):
        manual_outputs = pm.evaluate(data={"input": jax.numpy.ones((1, 1)), "running_input": manual_state})
        manual_state = manual_outputs["running_output"]


    # Automatically bind state inputs & outputs.
    model = Model()
    model |= Add()("local_input", "running_input", output="add_output")
    model |= Buffer()("add_output", output="local_output")
    # model.bind_state_input("running_input", "add_output", 1)

    main_model = Model()
    main_model |= model(local_input="input", local_output="output")

    pm = ml.compile(main_model, ml.JaxBackend(), jit=False, use_short_namings=False, inference=True, file_path="generated.py")
    state = pm.initial_state_dict
    for _ in range(10):
        outputs, state = pm.evaluate(data={"input": jax.numpy.ones((1, 1))}, state=state)

    assert jax.numpy.allclose(outputs["output"], manual_outputs["output"])

def test_running_mean_regular():
    import mithril as ml
    from mithril.models import Buffer, Square, Mean
    momentum = ml.IOKey("momentum")
    running_mean = ml.IOKey("running_mean")
    batch_mean = ml.IOKey("batch_mean")

    model = Model()
    model |= Mean()(input="input", output=batch_mean)
    model |= Buffer()((1 - momentum) * running_mean + momentum * batch_mean, output="new_mean")
    # model.bind_state_input("running_mean", "new_mean", 0)

    main_model = Model()
    main_model |= model(input="input", running_mean="running_mean", momentum=0.1, new_mean="mean")
    main_model |= Square()(input="mean", output=ml.IOKey("output"))

    pm = ml.compile(main_model, ml.JaxBackend(), jit=False, use_short_namings=False, inference=True, file_path="generated.py")
    res = pm.evaluate(data={"input": jax.numpy.ones((1, 1)), "running_mean": jax.numpy.zeros((1, 1))})
    ...