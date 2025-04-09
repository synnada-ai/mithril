# MLX Backend

This guide covers the MLX backend in Mithril, its features, configuration options, and usage patterns for Apple Silicon devices.

## Overview

The MLX backend enables Mithril models to run efficiently on Apple Silicon (M1/M2/M3) devices using Apple's MLX framework. MLX is a framework designed specifically for machine learning on Apple Silicon, leveraging the Metal Performance Shaders (MPS) and Apple Neural Engine (ANE) for accelerated computation.

## Features

The MLX backend provides:

- Optimized performance on Apple Silicon devices
- Automatic differentiation
- Lazy computation for optimization
- Memory-efficient operations
- Native Metal acceleration
- Just-in-time compilation

## Installation Requirements

To use the MLX backend, you need to install MLX:

```bash
pip install mlx
```

MLX requires macOS and an Apple Silicon device (M1, M2, or M3).

## Basic Usage

### Creating the Backend

```python
import mithril as ml
from mithril.backends import MLXBackend

# Create an MLX backend
backend = MLXBackend()

# Create and compile a model
model = ml.Model()
# ... define your model ...
compiled_model = ml.compile(model, backend)
```

### Configuration Options

You can configure the MLX backend with various options:

```python
import mlx.core as mx

backend = MLXBackend(
    dtype=mx.float16,                # Default data type
    device=mx.devices.gpu,           # Target device
    random_seed=42,                  # For reproducibility
    compute_precision="float32",     # Compute precision
    use_graph=True,                  # Enable computational graph
)
```

### Data Types

The MLX backend supports various data types:

```python
# Use float32 (default)
backend = MLXBackend(dtype=mx.float32)

# Use float16 for reduced memory usage
backend = MLXBackend(dtype=mx.float16)

# Use bfloat16 for better numerical stability than float16
backend = MLXBackend(dtype=mx.bfloat16)
```

## Device Management

### Selecting Compute Devices

MLX can utilize different device types:

```python
import mlx.core as mx

# Use GPU (Metal) - default
backend = MLXBackend(device=mx.devices.gpu)

# Use CPU
backend = MLXBackend(device=mx.devices.cpu)
```

### Memory Management

MLX is designed to be memory-efficient on Apple Silicon:

```python
# Configure memory handling
backend = MLXBackend(
    memory_limit="4GB",         # Limit total memory usage
    auto_free_tensors=True,     # Automatically free unused tensors
)
```

## Tensor Operations

### Creating Tensors

Creating tensors with the MLX backend:

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

### Tensor Conversion

Converting tensors between frameworks:

```python
# Convert from MLX array
import mlx.core as mx
mx_array = mx.random.normal((3, 4))
x = backend.from_mlx(mx_array)

# Convert to MLX array
mx_array = backend.to_mlx(x)

# Convert from NumPy
np_array = np.random.randn(3, 4)
x = backend.array(np_array)

# Convert to NumPy
np_array = backend.to_numpy(x)
```

## Automatic Differentiation

The MLX backend supports automatic differentiation:

```python
# Forward pass
outputs = compiled_model.evaluate(params, {"input": inputs})

# Compute loss
loss = compute_loss(outputs["output"], targets)

# Backward pass
outputs, gradients = compiled_model.evaluate(
    params, 
    {"input": inputs}, 
    output_gradients={"output": loss_gradients}
)
```

## Performance Optimization

### Lazy Evaluation

MLX uses lazy evaluation to optimize computation:

```python
# Operations are not executed immediately but recorded in a graph
x = backend.array([1.0, 2.0, 3.0])
y = backend.array([4.0, 5.0, 6.0])
z = backend.add(x, y)  # Not computed yet

# Computation happens when results are needed
result = backend.to_numpy(z)  # Triggers computation
```

### Computational Graphs

MLX uses computational graphs for optimization:

```python
# Enable graph mode for repeated operations
backend = MLXBackend(use_graph=True)

# Create a function that will be JIT compiled
@mx.compile
def forward_backward(params, inputs, targets):
    # Forward pass
    outputs = model_fn(params, inputs)
    
    # Loss computation
    loss = loss_fn(outputs, targets)
    
    # Backward pass
    grads = mx.grad(model_fn)(params, inputs)
    
    return loss, grads

# Use the compiled function
loss, grads = forward_backward(params, inputs, targets)
```

### Batching

For improved performance, use appropriate batch sizes:

```python
# MLX is efficient with moderately large batch sizes
batch_size = 32  # Experiment to find optimal size for your model
```

## Training with MLX Backend

### Basic Training Loop

```python
# Initialize parameters
params = compiled_model.get_parameters()

# Training loop
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in data_loader:
        # Forward pass and compute gradients
        outputs, gradients = compiled_model.evaluate(
            params,
            {"input": batch_inputs},
            output_gradients={"output": 2 * (outputs["output"] - batch_targets)}
        )
        
        # Update parameters
        for name, grad in gradients.items():
            params[name] = params[name] - learning_rate * grad
```

### Using MLX Optimizers

You can use MLX's built-in optimizers:

```python
import mlx.optimizers as optim

# Create an Adam optimizer
optimizer = optim.Adam(learning_rate=0.001)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = compiled_model.evaluate(params, {"input": inputs})
    
    # Compute loss and gradients
    loss = compute_loss(outputs["output"], targets)
    _, gradients = compiled_model.evaluate(
        params,
        {"input": inputs},
        output_gradients={"output": loss_gradients}
    )
    
    # Convert gradients to MLX format
    mlx_gradients = {name: backend.to_mlx(grad) for name, grad in gradients.items()}
    
    # Update parameters
    mlx_params = {name: backend.to_mlx(param) for name, param in params.items()}
    mlx_params = optimizer.update(mlx_params, mlx_gradients)
    
    # Convert back to backend format
    params = {name: backend.from_mlx(param) for name, param in mlx_params.items()}
```

## Advanced Features

### Custom MLX Operations

You can define custom operations:

```python
# Define a custom operation using MLX primitives
def custom_activation(x):
    # Implementation using MLX operations
    return backend.maximum(0.01 * x, x)

# Use in a model
model |= ml.Linear(dimension=64)(input="input", output="hidden")
model += ml.Lambda(custom_activation)(input="hidden", output="hidden_act")
model += ml.Linear(dimension=10)(input="hidden_act", output="output")
```

### Serialization

Saving and loading models with MLX backend:

```python
# Save parameters
import pickle

with open("model_params.pkl", "wb") as f:
    pickle.dump({k: backend.to_numpy(v) for k, v in params.items()}, f)

# Load parameters
with open("model_params.pkl", "rb") as f:
    loaded_params = pickle.load(f)
    params = {k: backend.array(v) for k, v in loaded_params.items()}
```

## Integration with Apple Ecosystem

### Metal Performance Shaders

MLX leverages Metal Performance Shaders for acceleration:

```python
# Enable MPS optimization
backend = MLXBackend(enable_mps_optimization=True)
```

### Apple Neural Engine

Some operations can leverage the Apple Neural Engine:

```python
# Enable ANE when available
backend = MLXBackend(use_ane=True)
```

### Memory Sharing with Other Frameworks

MLX can efficiently share memory with other Apple frameworks:

```python
# Create a tensor from a Core ML tensor without copying
from coremltools.models.ml_program import Tensor as MLTensor
ml_tensor = MLTensor(np_array)
x = backend.from_mlx(mx.array(ml_tensor))
```

## Best Practices

### Performance Optimization

1. **Use appropriate data types** for your task:
   ```python
   # Use float16 for inference
   backend = MLXBackend(dtype=mx.float16)
   
   # Use float32 for training
   backend = MLXBackend(dtype=mx.float32)
   ```

2. **Enable graph mode** for repeated operations:
   ```python
   backend = MLXBackend(use_graph=True)
   ```

3. **Batch operations** when possible:
   ```python
   # Process data in batches rather than one sample at a time
   ```

4. **Keep tensors on the same device** to avoid transfers:
   ```python
   # Avoid unnecessary device transfers
   ```

5. **Use compute_precision** appropriately:
   ```python
   # Higher precision for numerically sensitive operations
   backend = MLXBackend(compute_precision="float32")
   
   # Lower precision for better performance
   backend = MLXBackend(compute_precision="float16")
   ```

### Memory Management

1. **Release references** to unused large tensors:
   ```python
   # Explicitly delete tensors when no longer needed
   del large_tensor
   ```

2. **Monitor memory usage**:
   ```python
   # Check memory usage
   memory_info = backend.get_memory_info()
   print(f"Used memory: {memory_info['used']} MB")
   ```

## Debugging

### Common Debugging Techniques

1. **Enable verbose logging**:
   ```python
   compiled_model = ml.compile(model, backend, verbose=True)
   ```

2. **Print tensor information**:
   ```python
   print(f"Shape: {backend.shape(tensor)}, Dtype: {backend.dtype(tensor)}")
   ```

3. **Check for NaN values**:
   ```python
   has_nan = backend.isnan(tensor).any()
   print(f"Contains NaN: {has_nan}")
   ```

4. **Disable optimizations** for debugging:
   ```python
   backend = MLXBackend(use_graph=False)
   ```

## Common Issues and Solutions

### Issue: Out of Memory

**Solution**: Reduce batch size or model size, or use lower precision:
```python
# Use lower precision
backend = MLXBackend(dtype=mx.float16)

# Reduce batch size
```

### Issue: Slow Compilation

**Solution**: Simplify model or disable certain optimizations:
```python
# Disable graph optimization for faster compilation
backend = MLXBackend(use_graph=False)
```

### Issue: Numerical Instability

**Solution**: Use higher precision or normalize inputs:
```python
# Use higher precision
backend = MLXBackend(dtype=mx.float32, compute_precision="float32")

# Normalize inputs
inputs = (inputs - mean) / std
```

## Examples

### Basic Classification Model

```python
import mithril as ml
from mithril.backends import MLXBackend
import numpy as np

# Create a simple classification model
model = ml.Model()
model |= ml.Linear(dimension=128)(input="input", output="hidden1")
model += ml.Relu()(input="hidden1", output="hidden1_act")
model += ml.Linear(dimension=64)(input="hidden1_act", output="hidden2")
model += ml.Relu()(input="hidden2", output="hidden2_act")
model += ml.Linear(dimension=10)(input="hidden2_act", output="logits")
model += ml.Softmax()(input="logits", output="output")

# Compile with MLX backend
backend = MLXBackend()
compiled_model = ml.compile(model, backend)

# Generate synthetic data
X = np.random.randn(100, 784)
y = np.random.randint(0, 10, size=(100,))
y_one_hot = np.zeros((100, 10))
y_one_hot[np.arange(100), y] = 1

# Initialize parameters
params = compiled_model.get_parameters()

# Training loop
learning_rate = 0.01
for epoch in range(10):
    # Forward pass
    outputs = compiled_model.evaluate(params, {"input": X})
    predictions = outputs["output"]
    
    # Cross-entropy loss gradient
    loss_grad = predictions - y_one_hot
    
    # Backward pass
    _, gradients = compiled_model.evaluate(
        params, 
        {"input": X}, 
        output_gradients={"output": loss_grad}
    )
    
    # Update parameters
    for name, grad in gradients.items():
        params[name] = params[name] - learning_rate * grad
    
    # Calculate accuracy
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_classes == y)
    print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}")
```

### Image Processing Model

```python
# Create a convolutional model for image processing
model = ml.Model()
model |= ml.Conv2d(filters=16, kernel_size=3, stride=1, padding="same")(
    input="input", output="conv1"
)
model += ml.Relu()(input="conv1", output="conv1_act")
model += ml.MaxPool2d(kernel_size=2)(input="conv1_act", output="pool1")
model += ml.Conv2d(filters=32, kernel_size=3, stride=1, padding="same")(
    input="pool1", output="conv2"
)
model += ml.Relu()(input="conv2", output="conv2_act")
model += ml.MaxPool2d(kernel_size=2)(input="conv2_act", output="pool2")
model += ml.Flatten()(input="pool2", output="flat")
model += ml.Linear(dimension=64)(input="flat", output="hidden")
model += ml.Relu()(input="hidden", output="hidden_act")
model += ml.Linear(dimension=10)(input="hidden_act", output="output")

# Compile with MLX backend optimized for Apple Silicon
backend = MLXBackend(dtype=mx.float16)
compiled_model = ml.compile(model, backend)
```

## Compatibility Notes

### Hardware Compatibility

The MLX backend requires:
- macOS operating system
- Apple Silicon processor (M1, M2, or M3 family)

### Software Compatibility

- macOS 12 (Monterey) or newer
- Python 3.8 or newer
- MLX 0.0.3 or newer

### Comparison with Other Backends

| Feature | MLX | PyTorch | JAX |
|---------|-----|---------|-----|
| Apple Silicon Support | Native | Through MPS | Limited |
| Automatic Differentiation | Yes | Yes | Yes |
| Lazy Evaluation | Yes | No | Yes |
| JIT Compilation | Yes | Yes | Yes |
| Multi-GPU | Limited | Yes | Yes |
| ANE Support | Yes | No | No |

## Conclusion

The MLX backend provides excellent performance for Mithril models on Apple Silicon devices. By leveraging Apple's MLX framework, it offers native acceleration on M1/M2/M3 chips, with features like automatic differentiation and computational graph optimization. While it's specifically designed for Apple devices, it provides a powerful option for users in the Apple ecosystem to run machine learning models efficiently.