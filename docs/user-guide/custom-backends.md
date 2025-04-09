# Custom Backends

Mithril is designed to be backend-agnostic, allowing models to run on different computational frameworks. This guide explains how to create custom backends to extend Mithril's capabilities to new platforms and frameworks.

## Overview

A backend in Mithril is responsible for:

1. Providing tensor operations and data structures
2. Implementing core mathematical functions
3. Executing computation graphs
4. (Optionally) Supporting automatic differentiation

By implementing a custom backend, you can:

- Support a new computational framework
- Optimize for specific hardware
- Implement domain-specific optimizations
- Provide specialized functionality not available in standard backends

## Backend Interface

All backends must implement the `Backend` interface defined in `mithril.backends.backend.Backend`. This abstract base class defines the operations that every backend must support.

```python
from abc import ABC, abstractmethod

class Backend(ABC):
    @abstractmethod
    def tensor(self, data, dtype=None):
        """Create a tensor with the given data and type."""
        pass
    
    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create a tensor of zeros with the given shape and type."""
        pass
    
    # ... many more abstract methods
```

## Backend Categories

Mithril has two main categories of backends:

1. **Auto-differentiation Backends**: Support automatic computation of gradients (e.g., JAX, PyTorch, MLX)
2. **Manual-differentiation Backends**: Require manual gradient implementations (e.g., NumPy, C)

The choice of category depends on your target platform and requirements.

## Creating a Custom Backend

### Example: A Simple Custom Backend

Let's implement a simple custom backend that wraps NumPy:

```python
from mithril.backends.backend import Backend
import numpy as np

class MyCustomBackend(Backend):
    def __init__(self):
        super().__init__()
    
    def tensor(self, data, dtype=None):
        return np.array(data, dtype=self._convert_dtype(dtype))
    
    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=self._convert_dtype(dtype))
    
    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=self._convert_dtype(dtype))
    
    def _convert_dtype(self, dtype):
        # Convert Mithril dtype string to NumPy dtype
        if dtype is None:
            return None
        
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_
        }
        
        return dtype_map.get(dtype, None)
    
    # Implement other required methods...
```

### Required Operations

A complete backend implementation must implement all abstract methods from the `Backend` base class. These include:

1. **Tensor Creation**: `tensor`, `zeros`, `ones`, `eye`, `random_normal`, etc.
2. **Basic Arithmetic**: `add`, `subtract`, `multiply`, `divide`, etc.
3. **Advanced Math**: `matmul`, `exp`, `log`, `sin`, `cos`, etc.
4. **Shape Operations**: `reshape`, `transpose`, `expand_dims`, etc.
5. **Reduction Operations**: `reduce_sum`, `reduce_mean`, `reduce_max`, etc.
6. **NN Operations**: `conv2d`, `max_pool2d`, `batch_norm`, etc.

Refer to the existing backend implementations for examples.

## Implementing an Autograd Backend

If your backend supports automatic differentiation, you should inherit from a class in the `mithril.backends.with_autograd` package and implement additional gradient-related methods:

```python
from mithril.backends.with_autograd.backend import AutogradBackend

class MyAutogradBackend(AutogradBackend):
    def __init__(self):
        super().__init__()
    
    # Implement basic tensor operations...
    
    def grad(self, function):
        """Return a function that computes the gradient of `function`."""
        # Implement gradient computation
        pass
    
    def value_and_grad(self, function):
        """Return a function that computes both the output and gradients."""
        # Implement value and gradient computation
        pass
```

## Backend Organization

Custom backends should follow Mithril's organizational structure:

```
mithril/
  backends/
    with_autograd/
      my_autograd_backend/
        __init__.py
        backend.py        # Main backend implementation
        ops.py            # Optional: custom operations
        utils.py          # Optional: utility functions
    with_manualgrad/
      my_manual_backend/
        __init__.py
        backend.py
        ops.py
        utils.py
```

## Complete Example: Custom NumPy Backend with Extra Features

Here's a more complete example of a custom NumPy backend with additional features:

```python
from mithril.backends.with_manualgrad.numpy_backend import NumpyBackend
import numpy as np

class EnhancedNumpyBackend(NumpyBackend):
    """A NumPy backend with additional features for scientific computing."""
    
    def __init__(self, use_bfloat16=False):
        super().__init__()
        self.use_bfloat16 = use_bfloat16
        self._custom_functions = {}
    
    # Override tensor creation for custom behavior
    def tensor(self, data, dtype=None):
        # Handle bfloat16 emulation if enabled
        if dtype == "bfloat16" and self.use_bfloat16:
            # Emulate bfloat16 by truncating float32
            arr = np.array(data, dtype=np.float32)
            # Apply bfloat16 precision limitations
            arr = self._simulate_bfloat16(arr)
            return arr
        
        return super().tensor(data, dtype)
    
    def _simulate_bfloat16(self, arr):
        """Simulate bfloat16 precision by truncating float32."""
        # Convert to uint32 view
        as_uint32 = arr.view(np.uint32)
        # Clear the 16 least significant bits
        as_uint32 = as_uint32 & 0xFFFF0000
        # Convert back to float32
        return as_uint32.view(np.float32)
    
    # Add new scientific computing operations
    def fft(self, x):
        """Compute the Fast Fourier Transform of a tensor."""
        return np.fft.fft(x)
    
    def ifft(self, x):
        """Compute the Inverse Fast Fourier Transform of a tensor."""
        return np.fft.ifft(x)
    
    def solve(self, a, b):
        """Solve the linear system a*x = b for x."""
        return np.linalg.solve(a, b)
    
    # Support for custom functions
    def register_function(self, name, function):
        """Register a custom function with the backend."""
        self._custom_functions[name] = function
    
    def call_custom(self, name, *args, **kwargs):
        """Call a registered custom function."""
        if name not in self._custom_functions:
            raise ValueError(f"Custom function '{name}' not registered")
        
        return self._custom_functions[name](*args, **kwargs)
```

## Specialized Hardware Support

To support specialized hardware, you'll often need to integrate with low-level libraries:

```python
class TPUBackend(Backend):
    """A backend that targets Google TPUs."""
    
    def __init__(self):
        super().__init__()
        try:
            import jax
            import jax.numpy as jnp
            from jax.config import config
            
            # Configure JAX for TPU
            config.update("jax_xla_backend", "tpu_driver")
            config.update("jax_backend_target", "grpc://localhost:8470")
            
            self.jax = jax
            self.jnp = jnp
        except ImportError:
            raise ImportError(
                "JAX is required for TPUBackend. "
                "Please install jax and jaxlib."
            )
    
    # Implement backend methods using JAX/TPU...
```

## Testing Your Backend

It's crucial to thoroughly test your backend. Mithril provides test utilities for verifying backend compatibility:

```python
from mithril.testing.backend_tests import run_backend_tests

# Create an instance of your backend
my_backend = MyCustomBackend()

# Run the standard backend test suite
results = run_backend_tests(my_backend)
print(f"Passed: {results.passed}/{results.total} tests")
```

## Registering Your Backend

Once implemented, you can register your backend for easier access:

```python
from mithril.backends.registry import register_backend

# Register your backend
register_backend("my_custom", MyCustomBackend)

# Later, users can create it by name
from mithril.backends import get_backend
backend = get_backend("my_custom")
```

## Best Practices

1. **Start by Extending**: Rather than implementing from scratch, extend an existing backend that's closest to your target
2. **Prioritize Core Operations**: Implement the most commonly used operations first
3. **Test Thoroughly**: Verify your backend against the test suite and real-world models
4. **Handle Errors Gracefully**: Provide clear error messages when operations aren't supported
5. **Document Limitations**: Clearly document any operations that aren't supported or behave differently

## Common Challenges

### Type Conversion

Different frameworks handle data types differently. Create a mapping between Mithril's type strings and your framework's types:

```python
def _convert_dtype(self, dtype):
    if dtype is None:
        return None
    
    # Map from Mithril dtype strings to framework-specific types
    dtype_map = {
        "float32": framework.float32,
        "float64": framework.float64,
        "int32": framework.int32,
        "int64": framework.int64,
        # ...
    }
    
    return dtype_map.get(dtype)
```

### Shape Handling

Frameworks may have different shape conventions (e.g., channels-first vs. channels-last):

```python
def conv2d(self, x, filters, kernel_size, strides=(1, 1), padding="SAME"):
    # Convert from Mithril's NCHW format to framework's NHWC format
    x = self.transpose(x, (0, 2, 3, 1))
    
    # Perform convolution using framework's convention
    result = framework.conv2d(x, filters, kernel_size, strides, padding)
    
    # Convert back to Mithril's format
    return self.transpose(result, (0, 3, 1, 2))
```

### Gradient Calculation

If implementing a manual differentiation backend, you'll need to provide gradient calculations for each operation:

```python
def backward_tanh(self, grad_output, inputs, outputs):
    # Gradient of tanh is (1 - tanh(x)^2) * grad_output
    x = inputs[0]
    tanh_x = outputs[0]
    return grad_output * (1 - tanh_x * tanh_x)
```

## Examples of Production-Ready Backends

For reference, study these well-implemented backends in Mithril:

1. **JAX Backend**: `mithril/backends/with_autograd/jax_backend/`
2. **PyTorch Backend**: `mithril/backends/with_autograd/torch_backend/`
3. **NumPy Backend**: `mithril/backends/with_manualgrad/numpy_backend/`

## Conclusion

Creating a custom backend allows you to extend Mithril to support new platforms and frameworks. By following the structured approach outlined in this guide, you can implement backends that integrate seamlessly with the rest of the Mithril ecosystem, enabling users to run their models on your custom platform without changing their model code.