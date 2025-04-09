# Parallelization

Mithril provides a unified API for model and data parallelism across supporting frameworks. This guide covers how to set up and use parallelization features in Mithril.

## Overview

Parallelization in Mithril allows you to:

- Distribute model training and inference across multiple devices
- Parallelize any dimension of any input
- Use the same API regardless of the underlying framework
- Scale to large models and datasets efficiently

Currently, parallelization is supported for the PyTorch and JAX backends.

## Device Mesh

The foundation of parallelization in Mithril is the device mesh, which represents a logical arrangement of physical devices.

### Creating a Device Mesh

```python
import mithril as ml

# Create a 2D device mesh (2x2 = 4 devices)
backend = ml.JaxBackend(device_mesh=(2, 2))

# Create a 1D device mesh (4 devices in a row)
backend = ml.TorchBackend(device_mesh=(4,))
```

## Sharded Tensors

Sharded tensors are distributed across devices according to a sharding specification.

### Creating Sharded Tensors

```python
# Create a tensor sharded along the first dimension
# This will distribute the first dimension across 2 devices
sharded_tensor = backend.ones(128, 64, device_mesh=(2,))

# Create a tensor sharded along both dimensions
# This will distribute the tensor in a 2x2 grid
sharded_tensor = backend.ones(128, 64, device_mesh=(2, 2))
```

### Specifying Sharding

You can specify which tensor dimensions to shard:

```python
# Shard the first dimension only
sharded_tensor = backend.ones(128, 64, device_mesh=(2,), tensor_split=(0,))

# Shard the second dimension only
sharded_tensor = backend.ones(128, 64, device_mesh=(2,), tensor_split=(1,))

# Shard both dimensions across a 2D mesh
sharded_tensor = backend.ones(128, 64, device_mesh=(2, 2), tensor_split=(0, 1))
```

## Model Parallelism

Model parallelism involves partitioning the model itself across devices.

### Example: Parallelizing a Large Linear Layer

```python
import mithril as ml
from mithril.models import Linear

# Create a large linear layer
model = Linear(dimension=8192)

# Create a backend with a device mesh
backend = ml.TorchBackend(device_mesh=(2,))

# Compile the model
compiled_model = ml.compile(model, backend)

# Generate sharded parameters (weight will be sharded)
params = {"weight": backend.ones(8192, 4096, device_mesh=(2,)),
          "bias": backend.ones(8192)}

# Create input data (not sharded in this example)
inputs = {"input": backend.ones(32, 4096)}

# Run the parallelized model
outputs = compiled_model.evaluate(params, inputs)
```

## Data Parallelism

Data parallelism involves partitioning the data across devices while keeping model replicas on each device.

### Example: Data-parallel Training

```python
import mithril as ml
from mithril.models import Linear

# Create a model
model = Linear(dimension=128)

# Create a backend with a device mesh
backend = ml.TorchBackend(device_mesh=(2,))

# Compile the model
compiled_model = ml.compile(model, backend)

# Generate parameters (replicated across devices)
params = compiled_model.randomize_params()

# Create sharded input data (sharded along batch dimension)
batch_size = 128
inputs = {"input": backend.ones(batch_size, 256, device_mesh=(2,), tensor_split=(0,))}

# Forward pass
outputs, gradients = compiled_model.evaluate(
    params, 
    inputs, 
    output_gradients={"output": backend.ones(batch_size, 128, device_mesh=(2,), tensor_split=(0,))}
)
```

## Mixed Parallelism

You can combine model and data parallelism for greater flexibility.

### Example: Tensor Parallelism with Data Parallelism

```python
import mithril as ml
from mithril.models import Linear

# Create a large linear layer
model = Linear(dimension=8192)

# Create a backend with a 2D device mesh (4 devices total)
backend = ml.TorchBackend(device_mesh=(2, 2))

# Compile the model
compiled_model = ml.compile(model, backend)

# Generate sharded parameters
# - Weight is sharded along output dimension (dim 0) by the first mesh dimension
params = {"weight": backend.ones(8192, 4096, device_mesh=(2, 1), tensor_split=(0, None)),
          "bias": backend.ones(8192, device_mesh=(2, 1), tensor_split=(0, None))}

# Create sharded input data
# - Input is sharded along batch dimension (dim 0) by the second mesh dimension
batch_size = 128
inputs = {"input": backend.ones(batch_size, 4096, device_mesh=(1, 2), tensor_split=(None, 0))}

# Run the doubly-parallelized model
outputs = compiled_model.evaluate(params, inputs)
```

## Framework-Specific Details

### JAX Backend

For JAX, Mithril's parallelization maps to JAX's SPMD programming model:

```python
# JAX-specific setup
backend = ml.JaxBackend(device_mesh=(2, 2))

# This will use JAX's pjit and sharding under the hood
```

### PyTorch Backend

For PyTorch, Mithril uses PyTorch's distributed data-parallel and tensor-parallel features:

```python
# PyTorch-specific setup
backend = ml.TorchBackend(device_mesh=(2, 2))

# This will use PyTorch's DistributedDataParallel and tensor parallelism under the hood
```

## Best Practices

1. **Choose the right parallelism strategy**:
   - Data parallelism for compute-bound models with small parameters
   - Model parallelism for memory-bound models with large parameters
   - Mixed parallelism for very large models and datasets

2. **Balance device load**:
   - Distribute computation evenly across devices
   - Avoid communication bottlenecks

3. **Consider communication costs**:
   - Minimize cross-device communication
   - Group operations to reduce synchronization points

4. **Start simple**:
   - Begin with simpler parallelism strategies
   - Add complexity as needed

5. **Test thoroughly**:
   - Verify correctness with small-scale tests
   - Scale gradually to larger configurations