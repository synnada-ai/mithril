# NumPy Backend

This guide covers the NumPy backend in Mithril, including its features, capabilities, limitations, and common usage patterns.

## Overview

The NumPy backend provides a lightweight, CPU-based execution environment for Mithril models. It's based on the popular NumPy library, which offers efficient numerical computing capabilities in Python. This backend is ideal for smaller models, prototyping, and environments where GPU acceleration isn't available or necessary.

## Features

The NumPy backend offers:

- Pure Python implementation with minimal dependencies
- CPU-based computation
- Support for all standard Mithril operations
- Manual gradient calculation
- Seamless integration with NumPy's ecosystem
- Excellent debugging capabilities

## Installation Requirements

The NumPy backend requires only NumPy:

```bash
pip install numpy
```

The backend is automatically available when you install Mithril, as NumPy is a core dependency.

## Basic Usage

### Creating the Backend

```python
import mithril as ml
from mithril.backends import NumpyBackend

# Create a NumPy backend
backend = NumpyBackend()

# Create and compile a model
model = ml.Model()
# ... define your model ...
compiled_model = ml.compile(model, backend)
```

### Configuration Options

You can configure the NumPy backend with various options:

```python
backend = NumpyBackend(
    dtype=np.float64,           # Default data type
    randomness_seed=42,         # For deterministic behavior
    enable_optimizations=True,  # Enable NumPy-specific optimizations
)
```

### Data Types

The NumPy backend supports all NumPy data types:

```python
# Use float64 for high precision (default)
backend = NumpyBackend(dtype=np.float64)

# Use float32 for better performance and lower memory usage
backend = NumpyBackend(dtype=np.float32)

# Use int types for quantized operations
backend = NumpyBackend(dtype=np.int8)
```

## Tensor Operations

### Creating Tensors

Creating tensors with the NumPy backend:

```python
# Create from Python list
x = backend.array([[1, 2, 3], [4, 5, 6]])

# Create from NumPy array
import numpy as np
x_np = np.random.randn(3, 4)
x = backend.array(x_np)

# Create with specific shape
zeros = backend.zeros(3, 4)
ones = backend.ones(3, 4)
rand = backend.randn(3, 4)
```

### Basic Operations

```python
# Addition
c = backend.add(a, b)

# Subtraction
c = backend.subtract(a, b)

# Multiplication (element-wise)
c = backend.multiply(a, b)

# Division
c = backend.divide(a, b)

# Matrix multiplication
c = backend.matmul(a, b)
```

### Shape Operations

```python
# Reshape
b = backend.reshape(a, (2, 6))

# Transpose
b = backend.transpose(a)

# Concatenate
c = backend.concatenate([a, b], axis=0)

# Split
parts = backend.split(a, indices_or_sections=2, axis=0)
```

### Reduction Operations

```python
# Sum
total = backend.sum(a, axis=0)

# Mean
average = backend.mean(a, axis=1)

# Max/Min
maximum = backend.max(a)
minimum = backend.min(a)
```

## Manual Gradients

Unlike backends with automatic differentiation (JAX, PyTorch), the NumPy backend requires manual gradient calculation for training. Mithril provides built-in gradients for common operations:

```python
# Forward pass
outputs = compiled_model.evaluate(params, {"input": inputs})

# Compute loss gradients manually
output_gradients = {"output": 2 * (outputs["output"] - targets)}

# Backward pass with manual gradients
outputs, gradients = compiled_model.evaluate(
    params, 
    {"input": inputs}, 
    output_gradients=output_gradients
)

# Update parameters
for name, grad in gradients.items():
    params[name] = params[name] - learning_rate * grad
```

## Training with NumPy Backend

### Simple Training Loop

```python
# Initialize parameters
params = compiled_model.get_parameters()

# Training loop
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in data_loader:
        # Forward pass and compute loss gradients
        outputs, gradients = compiled_model.evaluate(
            params,
            {"input": batch_inputs},
            output_gradients={"output": 2 * (outputs["output"] - batch_targets)}
        )
        
        # Manual gradient descent update
        for name, grad in gradients.items():
            params[name] = params[name] - learning_rate * grad
```

### Custom Optimizers

You can implement custom optimizers for the NumPy backend:

```python
class SGDOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, params, gradients):
        for name, grad in gradients.items():
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(grad)
                
            self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * grad
            params[name] = params[name] + self.velocity[name]
        
        return params

# Using the optimizer
optimizer = SGDOptimizer(learning_rate=0.01)

# Training loop
for epoch in range(num_epochs):
    outputs, gradients = compiled_model.evaluate(
        params, 
        {"input": inputs}, 
        output_gradients=output_gradients
    )
    
    params = optimizer.update(params, gradients)
```

## Performance Considerations

### Optimizing Computation

The NumPy backend is generally slower than GPU-accelerated backends but can be optimized:

1. **Use the right data type** for your task:
   ```python
   # Use float32 for most tasks
   backend = NumpyBackend(dtype=np.float32)
   ```

2. **Vectorize operations** instead of using loops:
   ```python
   # Good: Vectorized operations
   result = backend.add(a, backend.multiply(b, c))
   
   # Avoid: Looping over elements
   ```

3. **Batch processing** for more efficient computation:
   ```python
   # Process data in larger batches when possible
   ```

4. **Enable optimizations**:
   ```python
   backend = NumpyBackend(enable_optimizations=True)
   ```

### Memory Management

Memory usage with the NumPy backend:

1. **Use appropriate data types** to reduce memory usage:
   ```python
   # Use smaller data types where precision isn't critical
   backend = NumpyBackend(dtype=np.float16)
   ```

2. **Release references** to unused tensors:
   ```python
   # Explicitly delete large tensors when no longer needed
   del large_tensor
   ```

3. **Use in-place operations** where available:
   ```python
   # Some operations support in-place updates
   backend.add_(a, b)  # In-place addition
   ```

## Debugging

The NumPy backend is particularly well-suited for debugging due to its simplicity:

1. **Inspect tensors** easily:
   ```python
   # Tensors are just NumPy arrays
   print(tensor)
   print(tensor.shape, tensor.dtype)
   print(np.min(tensor), np.max(tensor))
   ```

2. **Visualize data**:
   ```python
   import matplotlib.pyplot as plt
   
   # Plot activations
   plt.imshow(activation.reshape(8, 8))
   plt.colorbar()
   plt.show()
   ```

3. **Check for numerical issues**:
   ```python
   # Check for NaN or inf values
   print(np.isnan(tensor).any())
   print(np.isinf(tensor).any())
   ```

4. **Enable verbose logging**:
   ```python
   compiled_model = ml.compile(model, backend, verbose=True)
   ```

## Use Cases

### When to Use the NumPy Backend

The NumPy backend is ideal for:

1. **Prototyping and development**:
   - Simple, readable code for developing models
   - No GPU dependencies for rapid iteration

2. **Small models**:
   - Models that don't benefit significantly from GPU acceleration
   - When working with small datasets

3. **Educational purposes**:
   - Learning how models work from the ground up
   - Understanding gradient computation

4. **CPU-only environments**:
   - When GPUs are unavailable
   - In containerized or restricted environments

5. **Debugging**:
   - Tracking down numerical issues
   - Testing mathematical correctness

### When to Consider Other Backends

Consider alternatives when:

- Training large deep learning models (use JAX or PyTorch)
- Working with large datasets that benefit from GPU acceleration
- Requiring fast inference in production settings
- Needing automatic differentiation for complex models

## Integration with NumPy Ecosystem

### Using with NumPy Functions

You can seamlessly integrate with the broader NumPy ecosystem:

```python
# Convert Mithril tensor to raw NumPy array
raw_array = backend.to_numpy(tensor)

# Apply NumPy functions
processed = np.fft.fft2(raw_array)

# Convert back to Mithril tensor
result = backend.array(processed)
```

### Integration with SciPy

```python
from scipy import ndimage, optimize

# Process with SciPy functions
smoothed = ndimage.gaussian_filter(backend.to_numpy(image), sigma=2)
smoothed_tensor = backend.array(smoothed)

# Optimization with SciPy
def objective(params):
    params_dict = {name: backend.array(p) for name, p in zip(param_names, params)}
    outputs = compiled_model.evaluate(params_dict, {"input": inputs})
    return backend.to_numpy(outputs["loss"]).item()

result = optimize.minimize(objective, initial_params)
```

## Numerical Precision

### Managing Numerical Stability

The NumPy backend may experience numerical stability issues in some cases:

1. **Use higher precision** for sensitive calculations:
   ```python
   backend = NumpyBackend(dtype=np.float64)
   ```

2. **Apply numerical stabilization techniques**:
   ```python
   # For log-sum-exp, use the max trick
   def logsumexp(x, axis=None):
       x_max = np.max(x, axis=axis, keepdims=True)
       return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis))
   ```

3. **Normalize inputs** to avoid extreme values:
   ```python
   # Standardize inputs
   inputs = (inputs - np.mean(inputs)) / np.std(inputs)
   ```

## Limitations

### Known Limitations

The NumPy backend has several limitations compared to other backends:

1. **Performance**: Slower than GPU-accelerated backends for large computations
2. **Manual gradients**: No automatic differentiation
3. **Limited parallelism**: No built-in multi-GPU support
4. **Memory efficiency**: Less memory-efficient for large models
5. **Advanced operations**: Some specialized deep learning operations may be slower or missing

## Examples

### Basic Model

```python
import mithril as ml
from mithril.backends import NumpyBackend
import numpy as np

# Create a simple linear regression model
model = ml.Model()
model |= ml.Linear(dimension=1)(input="input", output="output")

# Compile with NumPy backend
backend = NumpyBackend()
compiled_model = ml.compile(model, backend)

# Generate synthetic data
X = np.random.randn(100, 5)
y = X.dot(np.array([1.0, -0.5, 0.25, 0.1, 0.7])).reshape(-1, 1) + 0.1 * np.random.randn(100, 1)

# Initialize parameters
params = compiled_model.get_parameters()

# Training loop
learning_rate = 0.01
for epoch in range(100):
    outputs = compiled_model.evaluate(params, {"input": X})
    predictions = outputs["output"]
    
    # MSE loss gradient
    output_gradients = {"output": 2 * (predictions - y) / len(X)}
    
    # Backward pass
    _, gradients = compiled_model.evaluate(
        params, 
        {"input": X}, 
        output_gradients=output_gradients
    )
    
    # Update parameters
    for name, grad in gradients.items():
        params[name] = params[name] - learning_rate * grad
    
    # Calculate loss
    if epoch % 10 == 0:
        mse = np.mean((predictions - y) ** 2)
        print(f"Epoch {epoch}, MSE: {mse:.6f}")
```

### Multi-layer Neural Network

```python
# Create a multi-layer neural network
model = ml.Model()
model |= ml.Linear(dimension=32)(input="input", output="hidden1")
model += ml.Relu()(input="hidden1", output="hidden1_act")
model += ml.Linear(dimension=16)(input="hidden1_act", output="hidden2")
model += ml.Relu()(input="hidden2", output="hidden2_act")
model += ml.Linear(dimension=1)(input="hidden2_act", output="output")

# Compile with NumPy backend
backend = NumpyBackend()
compiled_model = ml.compile(model, backend)

# Training (similar to previous example)
```

## Conclusion

The NumPy backend provides a simple, accessible way to use Mithril models in CPU-only environments. While it lacks the performance of GPU-accelerated backends, it offers excellent debugging capabilities, seamless integration with the NumPy ecosystem, and a straightforward path for understanding model behavior. It's an ideal choice for prototyping, educational purposes, and situations where computational resources are limited.