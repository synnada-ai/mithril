# PyTorch Backend

This guide covers the PyTorch backend in Mithril, its features, configuration options, and best practices.

## Overview

The PyTorch backend allows Mithril models to be compiled and executed using PyTorch as the underlying execution engine. This enables seamless integration with the PyTorch ecosystem while maintaining Mithril's model composition benefits.

## Features

The PyTorch backend provides:

- Automatic differentiation via PyTorch's autograd system
- GPU acceleration via CUDA
- Integration with PyTorch's optimizers and tools
- Dynamic computation graph support
- Native tensor operations

## Installation Requirements

To use the PyTorch backend, ensure that PyTorch is installed:

```bash
# Basic installation with CPU support
pip install torch

# For GPU support (with CUDA 11.8)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

Mithril automatically detects PyTorch when it's installed and makes the backend available.

## Basic Usage

### Creating the Backend

```python
import mithril as ml
from mithril.backends import TorchBackend

# Create a PyTorch backend
backend = TorchBackend()

# Create and compile a model
model = ml.Model()
# ... define your model ...
compiled_model = ml.compile(model, backend)
```

### Configuration Options

You can configure the PyTorch backend with various options:

```python
backend = TorchBackend(
    dtype=torch.float16,           # Default data type
    device="cuda:0",               # Target device
    requires_grad=True,            # Enable autograd
    use_amp=True,                  # Enable automatic mixed precision
    jit_compile=True,              # Use TorchScript when possible
    deterministic=True,            # Make operations deterministic
    benchmark=True,                # Enable cuDNN benchmarking
)
```

### Setting the Default Device

You can specify which device to use (CPU, CUDA, MPS):

```python
# Use the first CUDA device
backend = TorchBackend(device="cuda:0")

# Use CPU
backend = TorchBackend(device="cpu")

# Use Apple Metal (MPS)
backend = TorchBackend(device="mps")
```

### Data Types

The PyTorch backend supports various data types:

```python
# Use float32 (default)
backend = TorchBackend(dtype=torch.float32)

# Use float16 for reduced memory usage
backend = TorchBackend(dtype=torch.float16)

# Use bfloat16 for better numerical stability than float16
backend = TorchBackend(dtype=torch.bfloat16)
```

## Advanced Features

### JIT Compilation

The PyTorch backend can use TorchScript for additional optimizations:

```python
backend = TorchBackend(jit_compile=True)
```

JIT compilation offers:
- Improved execution speed
- Optimizations like operator fusion
- Potential performance gains, especially for complex models

### Automatic Mixed Precision (AMP)

For faster training and reduced memory consumption:

```python
backend = TorchBackend(use_amp=True)
```

AMP automatically uses lower precision where safe while maintaining model accuracy.

### Deterministic Operations

For reproducible results:

```python
backend = TorchBackend(
    deterministic=True,
    benchmark=False
)
```

This disables non-deterministic algorithms and cuDNN benchmark mode to ensure results are reproducible across runs.

## Memory Management

### Controlling Memory Usage

The PyTorch backend provides options to control memory usage:

```python
backend = TorchBackend(
    memory_limit="4GB",           # Limit total memory usage
    memory_allocation="lazy",     # Allocate tensors only when needed
    release_unused=True,          # Release unused tensors
)
```

### Memory Profiling

You can obtain memory usage information:

```python
# Get current memory usage
memory_stats = backend.get_memory_stats()
print(f"Allocated: {memory_stats['allocated']} bytes")
print(f"Cached: {memory_stats['cached']} bytes")

# Register a memory hook
def memory_hook(device, alloc_size, free_size):
    print(f"Allocating {alloc_size} bytes on {device}")

backend.register_memory_hook(memory_hook)
```

## Tensor Operations

### Creating Tensors

Creating tensors with the PyTorch backend:

```python
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
# Convert from PyTorch tensor
import torch
pt_tensor = torch.randn(3, 4)
x = backend.from_torch(pt_tensor)

# Convert to PyTorch tensor
pt_tensor = backend.to_torch(x)
```

## Training with PyTorch Backend

### Basic Training Loop

```python
import torch.optim as optim

# Compile model with PyTorch backend
compiled_model = ml.compile(model, TorchBackend())

# Extract model parameters
params = compiled_model.get_parameters()

# Create optimizer
optimizer = optim.Adam([{"params": list(params.values())}], lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        inputs, targets = batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass and compute loss gradients
        outputs, grads = compiled_model.evaluate(
            params,
            {"input": inputs},
            output_gradients={"output": 2 * (outputs["output"] - targets)}
        )
        
        # Set gradients in optimizer
        for name, grad in grads.items():
            params[name].grad = backend.to_torch(grad)
        
        # Update parameters
        optimizer.step()
        
        # Update the parameters in our model
        for name in params:
            params[name] = backend.from_torch(params[name])
```

### Using PyTorch's Training Features

You can leverage PyTorch's rich ecosystem:

```python
# Use learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Use PyTorch's built-in loss functions
loss_fn = torch.nn.MSELoss()

# Calculate loss gradient
def compute_loss_gradient(output, target):
    loss = loss_fn(backend.to_torch(output), target)
    loss.backward()
    return backend.from_torch(output.grad)
```

## Integration with PyTorch Ecosystem

### Using with PyTorch Datasets and DataLoaders

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    # ... dataset implementation ...

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    inputs, targets = batch
    outputs = compiled_model.evaluate(params, {"input": backend.from_torch(inputs)})
```

### Integrating with PyTorch Modules

You can combine Mithril models with PyTorch modules:

```python
import torch.nn as nn

# PyTorch module
class TorchModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

# Use in Mithril processing pipeline
torch_module = TorchModule()
outputs = compiled_model.evaluate(params, {"input": inputs})
processed = torch_module(backend.to_torch(outputs["output"]))
final_output = backend.from_torch(processed)
```

## Best Practices

### Performance Optimization

1. **Use the right device**:
   ```python
   backend = TorchBackend(device="cuda:0" if torch.cuda.is_available() else "cpu")
   ```

2. **Enable JIT compilation** for complex models:
   ```python
   backend = TorchBackend(jit_compile=True)
   ```

3. **Use AMP** for faster training on modern GPUs:
   ```python
   backend = TorchBackend(use_amp=True)
   ```

4. **Batch processing** for better GPU utilization:
   ```python
   # Process in batches rather than individual samples
   ```

5. **Profile your model** to identify bottlenecks:
   ```python
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
   ) as prof:
       outputs = compiled_model.evaluate(params, inputs)
   
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   ```

### Memory Management

1. **Use the appropriate precision** for your task:
   ```python
   # Use lower precision for inference or when memory is constrained
   backend = TorchBackend(dtype=torch.float16)
   ```

2. **Clear cache** when memory usage is high:
   ```python
   backend.clear_cache()
   ```

3. **Use checkpointing** for large models:
   ```python
   compiled_model = ml.compile(model, backend, use_checkpointing=True)
   ```

## Debugging

### Debugging Tips

1. **Check tensor shapes** and types:
   ```python
   # Print information about intermediate tensors
   compiled_model = ml.compile(model, backend, debug=True)
   ```

2. **Disable optimizations** for simpler debugging:
   ```python
   compiled_model = ml.compile(model, backend, optimization_level="none")
   ```

3. **Enable eager mode** to debug step by step:
   ```python
   backend = TorchBackend(eager_mode=True, jit_compile=False)
   ```

4. **Print intermediate values**:
   ```python
   # Add debug hooks to print intermediate values
   compiled_model.register_hook("hidden", lambda x: print("Hidden activation:", x.shape))
   ```

## Common Issues and Solutions

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or model size, or use GPU memory optimization:
```python
backend = TorchBackend(
    device="cuda:0",
    memory_efficient=True,
    use_amp=True  # Use half precision to reduce memory usage
)
```

### Issue: Slow Compilation

**Solution**: Disable JIT or use caching:
```python
# Disable JIT for faster compilation
backend = TorchBackend(jit_compile=False)

# Enable caching
compiled_model = ml.compile(model, backend, cache=True)
```

### Issue: Numerical Instability

**Solution**: Use higher precision or normalize inputs:
```python
# Use higher precision
backend = TorchBackend(dtype=torch.float32)

# Normalize inputs
inputs = {"input": (inputs["input"] - mean) / std}
```

## Migration from Pure PyTorch

If you're transitioning from pure PyTorch to Mithril with the PyTorch backend:

### Equivalent Operations

| PyTorch | Mithril with PyTorch Backend |
|---------|------------------------------|
| `torch.nn.Linear(10, 5)` | `ml.Linear(dimension=5)(input="x", output="y")` |
| `torch.nn.ReLU()` | `ml.Relu()(input="x", output="y")` |
| `torch.nn.Conv2d(3, 64, 3)` | `ml.Conv2d(filters=64, kernel_size=3)(input="x", output="y")` |

### Converting PyTorch Models

```python
# Simple PyTorch model
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Equivalent Mithril model
def create_mithril_model():
    model = ml.Model()
    model |= ml.Linear(dimension=64)(input="input", output="hidden")
    model += ml.Relu()(input="hidden", output="hidden_act")
    model += ml.Linear(dimension=1)(input="hidden_act", output="output")
    return model
```

## Conclusion

The PyTorch backend provides a powerful way to leverage PyTorch's capabilities while using Mithril's flexible model composition system. By understanding the configuration options and integration points, you can effectively use this backend for both research and production applications.