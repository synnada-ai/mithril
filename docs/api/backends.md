# Backends API

The Backends API provides interfaces for working with different computational backends in Mithril. This document covers how to select, configure, and use different backends.

## Available Backends

Mithril supports the following backends:

### Automatic Differentiation Backends

```python
# JAX Backend
from mithril.backends.with_autograd.jax_backend import JaxBackend

# PyTorch Backend
from mithril.backends.with_autograd.torch_backend import TorchBackend

# MLX Backend
from mithril.backends.with_autograd.mlx_backend import MlxBackend
```

### Manual Differentiation Backends

```python
# NumPy Backend
from mithril.backends.with_manualgrad.numpy_backend import NumpyBackend

# C Backend
from mithril.backends.with_manualgrad.c_backend import CBackend

# GGML Backend
from mithril.backends.with_manualgrad.ggml_backend import GGMLBackend
```

## Backend Interface

All backends implement the common `Backend` interface defined in `mithril.backends.backend.Backend`:

```python
class Backend(ABC):
    @abstractmethod
    def tensor(self, data, dtype=None):
        """Create a tensor with the given data and type."""
        pass
    
    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create a tensor of zeros with the given shape and type."""
        pass
    
    @abstractmethod
    def ones(self, shape, dtype=None):
        """Create a tensor of ones with the given shape and type."""
        pass
    
    # ... other abstract methods
```

## Using Backends

### Backend Selection

```python
from mithril.backends.with_autograd.jax_backend import JaxBackend
from mithril.models import LogicalModel

# Create a logical model
model = LogicalModel()
# ... define your model ...

# Select a backend
backend = JaxBackend()

# Compile the model for the selected backend
physical_model = model.compile(backend)

# Use the compiled model
output = physical_model(input_data)
```

### Backend Configuration

Backends can be configured with specific options:

```python
# JAX Backend with XLA compilation and platform selection
from mithril.backends.with_autograd.jax_backend import JaxBackend

backend = JaxBackend(
    jit_compile=True,
    platform="gpu",  # Can be "cpu", "gpu", or "tpu"
    precision="float32"
)

# PyTorch Backend with CUDA and mixed precision
from mithril.backends.with_autograd.torch_backend import TorchBackend

backend = TorchBackend(
    device="cuda",
    use_amp=True,
    compile=True  # Use torch.compile
)

# MLX Backend with specific seed
from mithril.backends.with_autograd.mlx_backend import MlxBackend

backend = MlxBackend(
    seed=42,
    default_dtype="float16"
)
```

### Backend-Specific Features

Some backends offer specific features not available in others:

```python
# JAX-specific features
jax_backend = JaxBackend()
jax_backend.set_platform("gpu")
jax_backend.enable_x64()

# PyTorch-specific features
torch_backend = TorchBackend()
torch_backend.set_device("cuda:0")
torch_backend.enable_deterministic()

# MLX-specific features
mlx_backend = MlxBackend()
mlx_backend.set_default_device("gpu")
```

## Backend Operations

Backends implement standard operations that are mapped from the logical model:

```python
# Create tensors
x = backend.tensor([1, 2, 3], dtype="float32")
y = backend.tensor([4, 5, 6], dtype="float32")

# Perform operations
z = backend.add(x, y)        # Element-wise addition
w = backend.matmul(x, y)     # Matrix multiplication

# Apply non-linear functions
a = backend.relu(z)          # ReLU activation
b = backend.sigmoid(w)       # Sigmoid activation

# Reduce operations
c = backend.reduce_sum(a)    # Sum all elements
d = backend.reduce_mean(b)   # Average all elements
```

## Backend Utilities

Each backend provides utility functions for common tasks:

```python
# Convert between numpy and backend tensors
np_array = backend.to_numpy(tensor)
tensor = backend.from_numpy(np_array)

# Get tensor shape and type
shape = backend.shape(tensor)
dtype = backend.dtype(tensor)

# Random number generation
random_tensor = backend.random_normal(shape=(3, 4), mean=0.0, stddev=1.0)

# Set random seed for reproducibility
backend.set_seed(42)
```

## Automatic Differentiation

Autograd backends support automatic differentiation:

```python
# JAX example
from mithril.backends.with_autograd.jax_backend import JaxBackend

backend = JaxBackend()

def f(x):
    return backend.reduce_sum(backend.power(x, 2))

x = backend.tensor([1.0, 2.0, 3.0])

# Compute gradient
grad_f = backend.grad(f)
grad_x = grad_f(x)
print(backend.to_numpy(grad_x))  # [2. 4. 6.]
```

## Parallelization

Backends support various forms of parallelization:

```python
from mithril.backends.with_autograd.jax_backend.parallel import data_parallel

# Data parallelism
parallel_fn = data_parallel(fn, backend=jax_backend, num_devices=4)
results = parallel_fn(data_batch)

# Model parallelism (device placement)
placement_spec = {"layer1": 0, "layer2": 1}  # Assigns layers to devices
placed_model = backend.place_model(model, placement_spec)
```

## Custom Backends

You can create custom backends by implementing the `Backend` interface:

```python
from mithril.backends.backend import Backend

class MyCustomBackend(Backend):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize custom backend
    
    def tensor(self, data, dtype=None):
        # Implement tensor creation
        pass
    
    def zeros(self, shape, dtype=None):
        # Implement zeros creation
        pass
    
    # Implement other required methods
```

## Best Practices

1. **Backend Selection**: Choose the appropriate backend for your use case:
   - **JAX**: For high-performance ML research, especially on accelerators
   - **PyTorch**: For flexibility and wide ecosystem support
   - **MLX**: For Apple Silicon optimization
   - **NumPy**: For CPU-only simple computation
   - **C/GGML**: For deployment to resource-constrained environments

2. **Memory Management**: Be mindful of memory usage, especially on accelerators

3. **Reproducibility**: Set random seeds for deterministic results

4. **Backend Switching**: Design models to be backend-agnostic when possible

5. **Optimization**: Use backend-specific optimizations when needed for performance

## Examples

See the [examples directory](https://github.com/example/mithril/tree/main/examples) for complete examples of working with different backends.