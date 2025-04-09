# Backends

Backends in Mithril are execution environments that implement the operations defined in your logical models. This guide covers the available backends, their features, and how to use them effectively.

## Available Backends

Mithril supports several backends, each with its own strengths:

### JAX Backend

The JAX backend provides efficient, compiled operations with automatic differentiation, suitable for CPU, GPU, and TPU.

```python
import mithril as ml

# Create a JAX backend
backend = ml.JaxBackend(dtype=ml.float32)
```

### PyTorch Backend

The PyTorch backend offers dynamic computation graphs and extensive ecosystem support.

```python
# Create a PyTorch backend
backend = ml.TorchBackend(dtype=ml.float32)
```

### NumPy Backend

A pure NumPy backend, useful for CPU computations without automatic differentiation requirements.

```python
# Create a NumPy backend
backend = ml.NumpyBackend(dtype=ml.float32)
```

### MLX Backend

Optimized for Apple Silicon (M1/M2/M3 chips), offering efficient operation on MacBooks and other Apple devices.

```python
# Create an MLX backend
backend = ml.MLXBackend(dtype=ml.float32)
```

### Experimental Backends

#### C/GGML Backend

Low-level, high-performance backends for deployment scenarios (experimental).

```python
# Create a C backend
backend = ml.CBackend(dtype=ml.float32)

# Create a GGML backend
backend = ml.GGMLBackend(dtype=ml.float32)
```

## Backend Configuration

### Data Types

You can specify the default data type for a backend:

```python
# Create a backend with float64 precision
backend = ml.JaxBackend(dtype=ml.float64)

# Available data types
# ml.float16, ml.float32, ml.float64, ml.int32, ml.int64
```

### Device Placement

For backends that support multiple devices:

```python
# Create a backend that uses GPU
backend = ml.TorchBackend(device="cuda")

# Create a backend that uses a specific GPU
backend = ml.TorchBackend(device="cuda:0")

# Create a backend that uses CPU
backend = ml.TorchBackend(device="cpu")
```

### Device Mesh for Parallelism

For distributed training scenarios:

```python
# Create a backend with a 2x2 device mesh
backend = ml.JaxBackend(device_mesh=(2, 2))

# Create a backend with a 1D device mesh
backend = ml.TorchBackend(device_mesh=(4,))
```

## Working with Backends

### Creating Tensors

Backends provide methods to create and manipulate tensors:

```python
# Create tensors
zeros = backend.zeros(3, 3)
ones = backend.ones(3, 3)
random = backend.randn(3, 3)
arange = backend.arange(10)

# Create specific data types
float16_tensor = backend.ones(3, 3, dtype=ml.float16)
```

### Tensor Operations

Backends implement standard tensor operations:

```python
# Basic operations
a = backend.ones(3, 3)
b = backend.ones(3, 3)
c = backend.add(a, b)
d = backend.matmul(a, b)

# Element-wise operations
e = backend.relu(a)
f = backend.sigmoid(a)
```

### Copying Between Backends

You can convert tensors between backends:

```python
# Create a tensor in JAX backend
jax_backend = ml.JaxBackend(dtype=ml.float32)
jax_tensor = jax_backend.ones(3, 3)

# Create a PyTorch backend
torch_backend = ml.TorchBackend(dtype=ml.float32)

# Convert JAX tensor to PyTorch tensor
torch_tensor = torch_backend.from_numpy(jax_backend.to_numpy(jax_tensor))
```

## Backend-specific Features

### JAX-specific Features

```python
# Enable JIT compilation
backend = ml.JaxBackend(jit=True)

# Set JAX random seed
backend.set_seed(42)

# Get JAX device
device = backend.device
```

### PyTorch-specific Features

```python
# Set PyTorch random seed
backend.set_seed(42)

# Move tensors between devices
cpu_tensor = backend.ones(3, 3, device="cpu")
gpu_tensor = backend.to_device(cpu_tensor, "cuda")
```

## Custom Backends

You can create custom backends by implementing the Backend interface:

```python
from mithril.backends import Backend

class MyCustomBackend(Backend):
    def __init__(self, dtype=ml.float32):
        super().__init__(dtype=dtype)
        # Initialize your backend
        
    def zeros(self, *shape, dtype=None):
        # Implement zeros
        
    def ones(self, *shape, dtype=None):
        # Implement ones
        
    # Implement other required methods
```

## Best Practices

1. **Match backend to hardware**: Choose JAX for TPUs, PyTorch for many GPUs, MLX for Apple Silicon
2. **Consistent data types**: Set the appropriate dtype for your backend to avoid precision issues
3. **Reuse backends**: Create a backend once and reuse it for multiple models
4. **Device placement**: Explicitly place operations on the correct device
5. **Backend compatibility**: Be aware of backend-specific behavior when switching between backends

## Limitations

Each backend has certain limitations:

- **NumPy Backend**: No automatic differentiation
- **C/GGML Backends**: Limited operation support, experimental
- **MLX Backend**: Only works on Apple Silicon devices

Be aware of these limitations when choosing a backend for your application.