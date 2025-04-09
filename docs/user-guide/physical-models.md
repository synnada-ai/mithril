# Physical Models

Physical models in Mithril represent the compiled form of logical models that are specific to a target backend. This document explains how physical models work and how to use them effectively.

## Overview

Physical models are the result of compiling logical models for a specific backend. They contain optimized representations of your model that are ready for execution on the target platform.

## Creating Physical Models

Physical models are typically created by compiling logical models:

```python
from mithril import LogicalModel
from mithril.backends.with_autograd.jax_backend import JaxBackend

# Create a logical model
logical_model = LogicalModel()
# ... define your model ...

# Compile to a physical model
backend = JaxBackend()
physical_model = logical_model.compile(backend)
```

## Physical Model Structure

A physical model contains:

- The compiled computation graph for the target backend
- Input and output specifications
- Optimized operators for the target platform
- Any necessary backend-specific data

## Key Features

### Execution Efficiency

Physical models are optimized for execution performance on the target backend:

```python
# Fast execution
output = physical_model(input_data)
```

### Memory Optimization

Physical models can be optimized to minimize memory usage during execution:

```python
# Create a memory-optimized physical model
physical_model = logical_model.compile(backend, memory_optimization=True)
```

### Backend-Specific Optimizations

Each backend has specific optimizations available for physical models:

- **JAX**: JIT compilation, XLA optimizations
- **PyTorch**: TorchScript, operator fusion
- **NumPy**: Vectorization, parallelization
- **MLX**: Apple Silicon optimizations

## Working with Physical Models

### Saving and Loading

Physical models can be saved and loaded for later use:

```python
# Save the compiled model
physical_model.save("my_model.mithril")

# Load the model later
from mithril.framework.physical import PhysicalModel
loaded_model = PhysicalModel.load("my_model.mithril", backend)
```

### Inference

Running inference with physical models:

```python
# Single input
output = physical_model(input_data)

# Batch processing
outputs = physical_model.batch_inference(input_batch)
```

### Training

Physical models can be used for training:

```python
from mithril.models.train_model import train

# Train the model
train(physical_model, train_data, optimizer, loss_fn, epochs=10)
```

## Best Practices

1. **Compile Once, Run Many Times**: Compilation can be expensive, so compile models once and reuse them for multiple executions.
2. **Backend Selection**: Choose the right backend for your target platform.
3. **Memory Management**: Use memory optimization for large models.
4. **Debugging**: Debug models at the logical level before compiling to physical models.

## Examples

See the [examples directory](https://github.com/example/mithril/tree/main/examples) for complete examples of working with physical models.