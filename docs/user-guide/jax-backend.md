# JAX Backend

This guide covers the JAX backend in Mithril, which provides support for high-performance computing using [JAX](https://jax.readthedocs.io/). JAX is a library for high-performance numerical computing with automatic differentiation and XLA compilation support.

## Overview

The JAX backend in Mithril enables:

- Fast computation on CPU, GPU, and TPU
- JIT (Just-In-Time) compilation for accelerated performance
- Automatic differentiation with gradient computation
- Parallel computation across multiple devices
- Functional programming patterns

## Setup

### Installation

To use the JAX backend, ensure you have JAX installed:

```bash
# CPU-only version
pip install jax

# GPU version (CUDA 11)
pip install "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# TPU version
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Creating a JAX Backend

```python
import mithril as ml

# Create a JAX backend with float32 default dtype
backend = ml.JaxBackend(dtype=ml.float32)

# Create a JAX backend with float64 precision
backend = ml.JaxBackend(dtype=ml.float64)

# Create a JAX backend with JIT enabled by default
backend = ml.JaxBackend(jit=True)

# Create a JAX backend for a specific device
backend = ml.JaxBackend(device="gpu")
```

## Basic Operations

### Creating Tensors

```python
# Create tensors of zeros and ones
zeros = backend.zeros(3, 4)
ones = backend.ones(3, 4)

# Create random tensors
randn = backend.randn(3, 4)  # Normal distribution
rand = backend.rand(3, 4)    # Uniform distribution

# Create tensors from NumPy arrays
import numpy as np
numpy_array = np.random.randn(3, 4)
jax_tensor = backend.array(numpy_array)

# Create arange
arange = backend.arange(10)

# Create tensors with specific data types
float16_tensor = backend.ones(3, 4, dtype=ml.float16)
int32_tensor = backend.ones(3, 4, dtype=ml.int32)
```

### Mathematical Operations

```python
# Basic math operations
a = backend.ones(3, 4)
b = backend.ones(3, 4)

add = backend.add(a, b)
sub = backend.subtract(a, b)
mul = backend.multiply(a, b)
div = backend.divide(a, b)

# Matrix operations
matmul = backend.matmul(backend.ones(3, 4), backend.ones(4, 5))

# Reductions
sum_all = backend.sum(a)
sum_axis = backend.sum(a, axis=0)
mean = backend.mean(a)
max_val = backend.max(a)

# Activations
relu = backend.relu(a)
sigmoid = backend.sigmoid(a)
tanh = backend.tanh(a)
```

### Shape Manipulation

```python
# Reshape
a = backend.ones(12)
reshaped = backend.reshape(a, (3, 4))

# Transpose
b = backend.ones(3, 4)
transposed = backend.transpose(b)

# Concatenate
c = backend.ones(3, 4)
d = backend.ones(3, 4)
concatenated = backend.concat([c, d], axis=0)  # Result shape: (6, 4)
```

## Compilation and Execution

### Compiling Models with JAX Backend

```python
import mithril as ml
from mithril.models import Linear

# Create a model
model = Linear(dimension=64)

# Create a JAX backend
backend = ml.JaxBackend(dtype=ml.float32)

# Compile the model
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 128]},  # Optional: specify input shapes
    jit=True  # Enable JIT compilation
)
```

### Running Inference

```python
# Generate random parameters
params = compiled_model.randomize_params()

# Create input data
inputs = {"input": backend.randn(32, 128)}

# Run inference
outputs = compiled_model.evaluate(params, inputs)
output_tensor = outputs["output"]
```

### Computing Gradients

```python
# Define output gradients (optional)
output_gradients = {"output": backend.ones_like(outputs["output"])}

# Compute outputs and gradients
outputs, gradients = compiled_model.evaluate(
    params, 
    inputs, 
    output_gradients=output_gradients
)

# Access gradients
weight_grad = gradients["weight"]
bias_grad = gradients["bias"]
```

## JIT Compilation

JIT compilation can significantly improve performance by optimizing the computation graph.

### Enabling JIT

```python
# Enable JIT during backend creation
backend = ml.JaxBackend(jit=True)

# Or enable it during compilation
compiled_model = ml.compile(
    model=model,
    backend=backend,
    jit=True
)
```

### JIT Compilation Benefits

- Faster execution through XLA optimization
- Fusion of operations for reduced memory usage
- Automatic parallelization across multiple cores

### JIT Compilation Limitations

- First call has compilation overhead
- Fixed shapes for input tensors (in basic mode)
- Limited support for dynamic control flow

## Parallelization

The JAX backend supports model and data parallelism for distributed training.

### Device Mesh

```python
# Create a 1D device mesh (2 devices)
backend = ml.JaxBackend(device_mesh=(2,))

# Create a 2D device mesh (2x2 = 4 devices)
backend = ml.JaxBackend(device_mesh=(2, 2))
```

### Sharded Tensors

```python
# Create a tensor sharded across 2 devices
sharded_tensor = backend.ones(128, 64, device_mesh=(2,))

# Create a tensor sharded along specific dimensions
sharded_tensor = backend.ones(128, 64, device_mesh=(2,), tensor_split=(0,))
```

### Parallel Evaluation

```python
# Create a large model
model = ml.models.MLP(input_size=1024, hidden_sizes=[2048, 2048], output_size=1024)

# Create a backend with a device mesh
backend = ml.JaxBackend(device_mesh=(2,))

# Compile the model
compiled_model = ml.compile(model, backend)

# Generate sharded parameters
params = compiled_model.randomize_params()

# Create sharded input data (sharded across batch dimension)
batch_size = 128
inputs = {"input": backend.ones(batch_size, 1024, device_mesh=(2,), tensor_split=(0,))}

# Run parallel evaluation
outputs = compiled_model.evaluate(params, inputs)
```

## Advanced Features

### Custom JAX Operations

You can register custom operations for the JAX backend:

```python
import jax
import jax.numpy as jnp

# Register a custom operation
@backend.register_op("gelu")
def gelu_jax(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

# Use in a model
from mithril.models import Model, Linear

model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")
model += "gelu"(input="hidden", output="hidden_act")  # Use custom op
model += Linear(dimension=10)(input="hidden_act", output="output")
```

### JAX Random Number Generation

```python
# Set random seed
backend.set_seed(42)

# Generate random numbers using JAX's PRNG system
key = backend.get_random_key()
new_key, subkey = backend.split_key(key)
random_numbers = backend.random_normal(subkey, shape=(3, 4))
```

### JAX Specific Options

```python
# Configure JAX backend options
backend = ml.JaxBackend(
    precision=jax.lax.Precision.HIGHEST,  # Set matmul precision
    debug=True,                           # Enable debugging features
    platform="gpu",                       # Specify platform
    jit=True                              # Enable JIT compilation
)
```

## Optimization with Optax

JAX doesn't include optimizers, but you can use [Optax](https://optax.readthedocs.io/) with Mithril's JAX backend:

```python
import optax

# Create a model and compile it
model = ml.models.MLP(input_size=784, hidden_sizes=[256, 128], output_size=10)
backend = ml.JaxBackend(dtype=ml.float32)
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.randomize_params()

# Create an optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    inputs = {"input": backend.array(X_batch)}
    outputs, gradients = compiled_model.evaluate(
        params, 
        inputs, 
        output_gradients={"output": compute_loss_gradient(outputs["output"], y_batch)}
    )
    
    # Update parameters with optax
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
```

## Comparison with Other Backends

### Advantages over PyTorch Backend

- Better performance through XLA compilation
- Excellent support for TPUs
- Functional programming style (immutable by default)
- More aggressive optimizations

### Disadvantages compared to PyTorch Backend

- Steeper learning curve
- Less flexible with dynamic shapes
- Smaller ecosystem of high-level tools
- Immutability can make some operations less intuitive

## Troubleshooting

### Common Issues and Solutions

1. **Memory Issues**:
   ```python
   # Use lower precision to reduce memory usage
   backend = ml.JaxBackend(dtype=ml.float16)
   ```

2. **Shape Errors**:
   ```python
   # Debug shape errors by compiling with verbose output
   compiled_model = ml.compile(model, backend, verbose=True)
   ```

3. **Performance Issues**:
   ```python
   # Ensure JIT is enabled
   compiled_model = ml.compile(model, backend, jit=True)
   ```

4. **Device Placement Issues**:
   ```python
   # Check which devices are available
   print(jax.devices())
   
   # Explicitly place operations on a device
   backend = ml.JaxBackend(device="gpu:0")
   ```

## Best Practices

1. **Minimize Host-Device Transfers**:
   - Keep data on the device as much as possible
   - Batch operations to reduce transfers

2. **Use the Right Precision**:
   - Use float16 or bfloat16 when possible (especially on TPUs)
   - Reserve float64 for operations that truly need it

3. **Leverage JIT Compilation**:
   - JIT-compile functions that are called multiple times
   - Keep the compiled function's interface stable

4. **Optimize for XLA**:
   - Use operations that XLA can optimize well
   - Avoid dynamic shapes when possible
   - Batch computations into large operations

5. **Parallel Processing**:
   - Use device meshes for multi-device computation
   - Consider both data and model parallelism strategies