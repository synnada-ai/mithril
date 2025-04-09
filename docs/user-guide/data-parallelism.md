# Data Parallelism

This guide explains how to use data parallelism in Mithril to accelerate training and inference across multiple devices.

## Overview

Data parallelism is a technique where the same model is replicated across multiple devices (GPUs, TPUs, etc.), with each device processing a different subset of the input data. This approach allows you to:

- Scale to larger batch sizes
- Accelerate training and inference
- Efficiently utilize multiple accelerators
- Maintain the same model architecture

## Basic Concept

In data parallelism:

1. The model is replicated on each device
2. The input data is split into chunks
3. Each device processes its chunk independently
4. Results are combined (e.g., gradients are averaged during training)

![Data Parallelism Diagram](../assets/data_parallelism.png)

## Implementing Data Parallelism in Mithril

Mithril provides built-in support for data parallelism through the device mesh and sharding features.

### Setting Up a Device Mesh

A device mesh represents the logical arrangement of your devices:

```python
import mithril as ml
from mithril.backends import JaxBackend, TorchBackend

# Create a backend with a linear device mesh across 4 GPUs
jax_backend = JaxBackend(device_mesh=(4,))

# For PyTorch backend
torch_backend = TorchBackend(device_mesh=(4,))
```

### Creating and Compiling a Model

Create and compile your model as usual:

```python
# Create a model
model = ml.Model()
model |= ml.Linear(dimension=128)(input="input", output="hidden")
model += ml.Relu()(input="hidden", output="hidden_act")
model += ml.Linear(dimension=10)(input="hidden_act", output="output")

# Compile with the multi-device backend
compiled_model = ml.compile(model, jax_backend)
```

### Data Sharding

To leverage data parallelism, you need to shard your input data across the first dimension (typically the batch dimension):

```python
# Create sharded input data across 4 GPUs
batch_size = 256  # Total batch size across all devices
inputs = {
    "input": jax_backend.randn(
        batch_size, 784,  # Shape: [batch_size, features]
        device_mesh=(4,),  # 4 devices in mesh
        tensor_split=(0,)  # Split along batch dimension (0)
    )
}
```

The `tensor_split=(0,)` parameter indicates that the tensor should be split along dimension 0 (the batch dimension). Each device will receive a batch_size/4 = 64 examples.

### Forward and Backward Pass

The forward and backward passes work the same as with a single device, but computations are performed in parallel:

```python
# Forward pass (performed in parallel across devices)
outputs = compiled_model.evaluate(params, inputs)

# For training, compute loss gradients
output_gradients = {"output": calculate_loss_gradient(outputs["output"], targets)}

# Backward pass (gradients are automatically averaged across devices)
outputs, gradients = compiled_model.evaluate(
    params, 
    inputs, 
    output_gradients=output_gradients
)
```

### Parameter Updates

When updating parameters, they remain replicated across devices:

```python
# Update parameters (performed on all devices)
for name, grad in gradients.items():
    params[name] = params[name] - learning_rate * grad
```

## Advanced Data Parallelism

### Gradient Accumulation

For very large models or to simulate larger batch sizes, you can use gradient accumulation:

```python
# Initialize accumulated gradients
accumulated_gradients = {name: jax_backend.zeros_like(param) for name, param in params.items()}
accumulation_steps = 4

# Training loop with gradient accumulation
for step in range(accumulation_steps):
    # Get batch for this step
    batch_inputs = get_batch(step)
    
    # Forward and backward pass
    outputs, step_gradients = compiled_model.evaluate(
        params, 
        batch_inputs, 
        output_gradients=calculate_loss_gradient(...)
    )
    
    # Accumulate gradients
    for name, grad in step_gradients.items():
        accumulated_gradients[name] = accumulated_gradients[name] + grad

# Scale accumulated gradients
for name in accumulated_gradients:
    accumulated_gradients[name] = accumulated_gradients[name] / accumulation_steps

# Update parameters with accumulated gradients
for name, grad in accumulated_gradients.items():
    params[name] = params[name] - learning_rate * grad
```

### Mixed Precision Training

You can combine data parallelism with mixed precision training for further acceleration:

```python
# Create a backend with mixed precision
jax_backend = JaxBackend(
    device_mesh=(4,),
    compute_dtype="bfloat16",  # Use bfloat16 for computation
    param_dtype="float32"      # Keep parameters in float32
)
```

### Optimizing Communication

Efficient communication between devices is crucial for data parallelism:

```python
# Create a backend with optimized communication
jax_backend = JaxBackend(
    device_mesh=(4,),
    collective_ops="nccl",            # Use NCCL for communication (for GPUs)
    collective_optimization=True       # Enable communication optimization
)
```

## Backend-Specific Data Parallelism

### JAX Backend

JAX has excellent support for data parallelism through its pmap and pjit functions:

```python
jax_backend = JaxBackend(
    device_mesh=(8,),  # 8 GPUs/TPUs
    pjit_mesh=True,    # Use pjit for partitioning
    spmd_mode=True     # Enable SPMD (Single Program Multiple Data) mode
)
```

### PyTorch Backend

PyTorch supports data parallelism through DistributedDataParallel:

```python
torch_backend = TorchBackend(
    device_mesh=(4,),                # 4 GPUs
    distributed_backend="nccl",      # Communication backend
    find_unused_parameters=False,    # Optimization flag
    gradient_as_bucket_view=True     # Reduce peak memory usage
)
```

### MLX Backend

For Apple Silicon devices, you can use data parallelism across multiple M-series chips:

```python
mlx_backend = MLXBackend(
    device_mesh=(2,),  # 2 M-series chips
    memory_format="channel_last"  # Optimized memory format for Apple hardware
)
```

## Performance Tuning

### Batch Size Selection

Choosing the right batch size is crucial for data parallelism:

```python
# Larger batch sizes typically benefit more from data parallelism
optimal_local_batch = 32  # Per-device batch size
num_devices = 4
global_batch_size = optimal_local_batch * num_devices  # 128
```

As a rule of thumb:
- Start with a batch size that fills device memory on a single device
- Scale this batch size linearly with the number of devices
- Adjust learning rate when scaling batch size (often using the "square root scaling rule")

### Communication Optimization

Reduce communication overhead with these techniques:

1. **Gradient Accumulation**: Reduce synchronization frequency
   ```python
   # Accumulate gradients over multiple steps
   accumulation_steps = 4
   ```

2. **Gradient Compression**: Reduce communication volume
   ```python
   # Enable gradient compression
   torch_backend = TorchBackend(
       device_mesh=(4,),
       gradient_compression="fp16"  # Compress gradients to fp16
   )
   ```

3. **Overlap Computation and Communication**: Hide communication latency
   ```python
   # Enable computation-communication overlap
   jax_backend = JaxBackend(
       device_mesh=(4,),
       async_collective_ops=True  # Asynchronous collectives
   )
   ```

## Monitoring and Debugging

### Profiling Performance

Monitor the performance of data-parallel training:

```python
# Enable profiling
with ml.profile(backend) as prof:
    outputs, gradients = compiled_model.evaluate(params, inputs, output_gradients)

# Print profiling results
print(prof.summary())
```

### Checking Device Utilization

Ensure all devices are efficiently utilized:

```python
# For JAX backend
from jax.experimental.maps import thread_resources
print(f"Devices used: {thread_resources.env.physical_mesh.devices}")

# For PyTorch backend
import torch
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
```

## Best Practices

1. **Scale batch size with number of devices**: Maintain the same per-device batch size
2. **Adjust learning rate**: When increasing global batch size, adjust learning rate accordingly
3. **Use mixed precision**: Combine data parallelism with mixed precision for maximum performance
4. **Minimize host-device transfers**: Keep data on devices as much as possible
5. **Use efficient data loading**: Parallelize data loading to keep devices fed
6. **Balance device workloads**: Ensure equal data distribution across devices
7. **Monitor device utilization**: Check that all devices are working efficiently

## Limitations

Data parallelism has some limitations:

1. **Communication overhead**: Performance scaling is not linear due to increasing communication
2. **Memory replication**: Parameters are replicated on each device, limiting model size
3. **Batch size constraints**: Very large batch sizes may affect convergence
4. **Load balancing challenges**: Uneven work distribution can lead to inefficiency

## Comparison with Other Parallelism Techniques

| Feature | Data Parallelism | Model Parallelism | Pipeline Parallelism |
|---------|------------------|-------------------|----------------------|
| Implementation Complexity | Low | High | Medium |
| Memory Efficiency | Low (replicated model) | High | Medium |
| Communication Cost | High (gradients) | Medium | Low |
| Load Balancing | Simple | Challenging | Moderate |
| Best For | Small to medium models | Very large models | Deep models |

## Examples

### Training a CNN with Data Parallelism

```python
import mithril as ml
from mithril.backends import TorchBackend
import numpy as np

# Create a CNN model
model = ml.Model()
model |= ml.Conv2d(filters=32, kernel_size=3)(input="input", output="conv1")
model += ml.Relu()(input="conv1", output="relu1")
model += ml.MaxPool2d(kernel_size=2)(input="relu1", output="pool1")
model += ml.Flatten()(input="pool1", output="flat")
model += ml.Linear(dimension=128)(input="flat", output="fc1")
model += ml.Relu()(input="fc1", output="relu2")
model += ml.Linear(dimension=10)(input="fc1", output="output")

# Create a backend with 4 GPUs
backend = TorchBackend(device_mesh=(4,))

# Compile model
compiled_model = ml.compile(model, backend)

# Create data-parallel inputs
batch_size = 256  # Global batch size
inputs = {
    "input": backend.randn(
        batch_size, 3, 32, 32,  # [batch, channels, height, width]
        device_mesh=(4,),
        tensor_split=(0,)  # Split along batch dimension
    )
}

# Create random targets
targets = backend.randint(0, 10, (batch_size,), device_mesh=(4,), tensor_split=(0,))

# Initialize parameters
params = compiled_model.get_parameters()

# Training loop
learning_rate = 0.01
for epoch in range(10):
    # Forward pass
    outputs = compiled_model.evaluate(params, inputs)
    
    # Compute loss gradient
    loss_gradient = compute_cross_entropy_gradient(outputs["output"], targets)
    
    # Backward pass
    _, gradients = compiled_model.evaluate(
        params, 
        inputs, 
        output_gradients={"output": loss_gradient}
    )
    
    # Update parameters
    for name, grad in gradients.items():
        params[name] = params[name] - learning_rate * grad
    
    # Compute accuracy
    predictions = backend.argmax(outputs["output"], axis=1)
    accuracy = backend.mean(backend.equal(predictions, targets))
    print(f"Epoch {epoch}, Accuracy: {backend.to_numpy(accuracy):.4f}")
```

## Conclusion

Data parallelism is a powerful technique for scaling Mithril models across multiple devices. By distributing the data and replicating the model, you can significantly accelerate training and inference while maintaining model accuracy. Mithril's device mesh and sharding capabilities make it straightforward to implement data-parallel training on various backends, from GPUs to TPUs and even Apple Silicon devices.