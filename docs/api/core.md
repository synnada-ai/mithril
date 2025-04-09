# Core API Reference

This page documents the core API of Mithril, including functions and classes that provide the foundation for the library.

## Main Functions

### compile

```python
def compile(model, backend, shapes=None, types=None, jit=None, file_path=None, verbose=False):
    """
    Compile a logical model into a physical model for a specific backend.

    Parameters:
        model: The logical model to compile
        backend: The backend to compile for (e.g., JaxBackend, TorchBackend)
        shapes: Optional dictionary mapping input terminal names to shapes
        types: Optional dictionary mapping input terminal names to data types
        jit: Whether to use JIT compilation (if available for the backend)
        file_path: Optional path to write the generated code to
        verbose: Whether to print verbose compilation information

    Returns:
        A compiled physical model
    """
```

### randomize_params

```python
def randomize_params(model, backend, shapes=None, types=None):
    """
    Generate random parameters for a model.

    Parameters:
        model: The logical model to generate parameters for
        backend: The backend to use for parameter generation
        shapes: Optional dictionary mapping input terminal names to shapes
        types: Optional dictionary mapping input terminal names to data types

    Returns:
        A dictionary of random parameters
    """
```

### evaluate

```python
def evaluate(model, params, inputs, output_gradients=None):
    """
    Evaluate a model with the given parameters and inputs.

    Parameters:
        model: The compiled physical model to evaluate
        params: The parameters to use for evaluation
        inputs: The inputs to the model
        output_gradients: Optional gradients of the loss with respect to outputs

    Returns:
        If output_gradients is None:
            A dictionary mapping output terminal names to output values
        Else:
            A tuple (outputs, gradients) where gradients is a dictionary
            mapping parameter names to gradient values
    """
```

## Data Types

Mithril defines the following data types:

```python
# Float types
float16  # 16-bit floating point
float32  # 32-bit floating point
float64  # 64-bit floating point

# Integer types
int8     # 8-bit signed integer
int16    # 16-bit signed integer
int32    # 32-bit signed integer
int64    # 64-bit signed integer
uint8    # 8-bit unsigned integer
uint16   # 16-bit unsigned integer
uint32   # 32-bit unsigned integer
uint64   # 64-bit unsigned integer

# Boolean type
bool     # Boolean values
```

## Backend Classes

Mithril provides several backend classes, each implementing the `Backend` interface:

```python
class Backend:
    """Base class for all backends."""
    
    def __init__(self, dtype=None, device=None, device_mesh=None):
        """
        Initialize a backend.
        
        Parameters:
            dtype: The default data type for the backend
            device: The device to use (e.g., "cpu", "cuda")
            device_mesh: Optional mesh of devices for parallelization
        """
    
    # Tensor creation methods
    def zeros(self, *shape, dtype=None):
        """Create a tensor of zeros with the given shape."""
        
    def ones(self, *shape, dtype=None):
        """Create a tensor of ones with the given shape."""
        
    def randn(self, *shape, dtype=None):
        """Create a tensor of random values from a normal distribution."""
        
    def array(self, data, dtype=None):
        """Create a tensor from the given data."""
        
    def arange(self, start, stop=None, step=1, dtype=None):
        """Create a tensor with evenly spaced values within a given interval."""
    
    # Mathematical operations
    def add(self, a, b):
        """Add two tensors element-wise."""
        
    def subtract(self, a, b):
        """Subtract two tensors element-wise."""
        
    def multiply(self, a, b):
        """Multiply two tensors element-wise."""
        
    def divide(self, a, b):
        """Divide two tensors element-wise."""
        
    def matmul(self, a, b):
        """Perform matrix multiplication of two tensors."""
    
    # Activation functions
    def relu(self, x):
        """Apply the ReLU function element-wise."""
        
    def sigmoid(self, x):
        """Apply the sigmoid function element-wise."""
        
    def tanh(self, x):
        """Apply the hyperbolic tangent function element-wise."""
    
    # Reduction operations
    def sum(self, x, axis=None, keepdims=False):
        """Sum tensor elements along specified axes."""
        
    def mean(self, x, axis=None, keepdims=False):
        """Compute the mean along specified axes."""
        
    def max(self, x, axis=None, keepdims=False):
        """Compute the maximum along specified axes."""
    
    # Shape manipulation
    def reshape(self, x, shape):
        """Reshape a tensor to the given shape."""
        
    def transpose(self, x, axes=None):
        """Permute the dimensions of a tensor."""
    
    # Utility functions
    def to_numpy(self, x):
        """Convert a tensor to a NumPy array."""
        
    def from_numpy(self, x):
        """Create a tensor from a NumPy array."""
```

### JaxBackend

```python
class JaxBackend(Backend):
    """Backend implementation using JAX."""
    
    def __init__(self, dtype=None, device=None, device_mesh=None, jit=False):
        """
        Initialize a JAX backend.
        
        Parameters:
            dtype: The default data type for the backend
            device: The device to use (e.g., "cpu", "gpu")
            device_mesh: Optional mesh of devices for parallelization
            jit: Whether to use JIT compilation by default
        """
```

### TorchBackend

```python
class TorchBackend(Backend):
    """Backend implementation using PyTorch."""
    
    def __init__(self, dtype=None, device=None, device_mesh=None):
        """
        Initialize a PyTorch backend.
        
        Parameters:
            dtype: The default data type for the backend
            device: The device to use (e.g., "cpu", "cuda")
            device_mesh: Optional mesh of devices for parallelization
        """
```

### NumpyBackend

```python
class NumpyBackend(Backend):
    """Backend implementation using NumPy."""
    
    def __init__(self, dtype=None):
        """
        Initialize a NumPy backend.
        
        Parameters:
            dtype: The default data type for the backend
        """
```

### MLXBackend

```python
class MLXBackend(Backend):
    """Backend implementation using MLX."""
    
    def __init__(self, dtype=None, device_mesh=None):
        """
        Initialize an MLX backend.
        
        Parameters:
            dtype: The default data type for the backend
            device_mesh: Optional mesh of devices for parallelization
        """
```

## Physical Model

```python
class PhysicalModel:
    """
    A compiled physical model that can be evaluated.
    """
    
    def evaluate(self, params, inputs, output_gradients=None):
        """
        Evaluate the model with the given parameters and inputs.
        
        Parameters:
            params: The parameters to use for evaluation
            inputs: The inputs to the model
            output_gradients: Optional gradients of the loss with respect to outputs
        
        Returns:
            If output_gradients is None:
                A dictionary mapping output terminal names to output values
            Else:
                A tuple (outputs, gradients) where gradients is a dictionary
                mapping parameter names to gradient values
        """
    
    def randomize_params(self):
        """
        Generate random parameters for the model.
        
        Returns:
            A dictionary of random parameters
        """
    
    def get_shapes(self):
        """
        Get the shapes of inputs, outputs, and parameters.
        
        Returns:
            A dictionary mapping terminal/parameter names to shapes
        """
    
    def get_types(self):
        """
        Get the data types of inputs, outputs, and parameters.
        
        Returns:
            A dictionary mapping terminal/parameter names to data types
        """
```