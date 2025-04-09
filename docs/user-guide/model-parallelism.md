# Model Parallelism

This guide explains how to use model parallelism in Mithril to distribute large models across multiple devices.

## Overview

Model parallelism is a technique where a single model is split across multiple devices (GPUs, TPUs, etc.), with each device handling different parts of the model. This approach is essential when:

- Your model is too large to fit on a single device's memory
- You want to accelerate computation by parallelizing model operations
- You need to optimize for specific hardware characteristics

Unlike data parallelism, which replicates the entire model and splits the data, model parallelism divides the model itself.

## Basic Concept

In model parallelism:

1. The model is partitioned into segments
2. Each segment is placed on a different device
3. Data flows through devices in sequence during forward and backward passes
4. Communication occurs between devices to transfer activations and gradients

![Model Parallelism Diagram](../assets/model_parallelism.png)

## Implementing Model Parallelism in Mithril

Mithril provides several mechanisms for implementing model parallelism, leveraging the device mesh and sharding capabilities.

### Setting Up a Device Mesh

First, create a device mesh to represent the logical arrangement of your devices:

```python
import mithril as ml
from mithril.backends import JaxBackend, TorchBackend

# Create a backend with a linear device mesh across 4 GPUs
jax_backend = JaxBackend(device_mesh=(4,))

# For PyTorch backend
torch_backend = TorchBackend(device_mesh=(4,))
```

### Partitioning Operators

You can specify which device each operator in your model should run on:

```python
# Create a model with operators assigned to different devices
model = ml.Model()
model |= ml.Linear(dimension=1024, device_id=0)(input="input", output="hidden1")
model += ml.Relu(device_id=0)(input="hidden1", output="hidden1_act")
model += ml.Linear(dimension=1024, device_id=1)(input="hidden1_act", output="hidden2")
model += ml.Relu(device_id=1)(input="hidden2", output="hidden2_act")
model += ml.Linear(dimension=1024, device_id=2)(input="hidden2_act", output="hidden3")
model += ml.Relu(device_id=2)(input="hidden3", output="hidden3_act")
model += ml.Linear(dimension=10, device_id=3)(input="hidden3_act", output="output")
```

### Parameter Sharding

For very large models, you can shard individual parameters across devices:

```python
# Create a large linear layer with weights sharded across 4 devices
model |= ml.Linear(
    dimension=16384, 
    weight_shard_spec=(0, None),  # Shard first dimension of weights
    device_mesh=(4,)              # Across 4 devices
)(input="input", output="hidden")
```

The `weight_shard_spec=(0, None)` parameter indicates that the weights should be sharded along dimension 0, with each device holding a portion of the input dimension.

### Compiling and Executing

Compile the model as usual:

```python
# Compile with the multi-device backend
compiled_model = ml.compile(model, jax_backend)
```

For execution, you typically don't need to shard the input data (unlike data parallelism):

```python
# Create inputs (not sharded since we're using model parallelism)
inputs = {"input": jax_backend.randn(32, 2048)}  # Batch size 32, 2048 features

# Forward pass
outputs = compiled_model.evaluate(params, inputs)
```

## Advanced Model Parallelism

### Tensor Parallelism

Tensor parallelism splits individual tensors across devices, particularly useful for large matrix multiplications:

```python
# Create a model with tensor parallelism
model = ml.Model()
model |= ml.Linear(
    dimension=16384,
    weight_shard_spec=(0, 1),  # Shard both dimensions of weights
    device_mesh=(2, 2)         # 2x2 device mesh (4 devices total)
)(input="input", output="hidden")
model += ml.Relu()(input="hidden", output="hidden_act")
model += ml.Linear(
    dimension=4096,
    weight_shard_spec=(0, None),  # Shard only input dimension
    device_mesh=(2, 2)
)(input="hidden_act", output="output")
```

In this example, we're using a 2D device mesh (2×2) to shard the weights in both dimensions. This enables more efficient computation of the matrix multiplication operation.

### Pipeline Parallelism

Pipeline parallelism divides the model into sequential stages, with each stage on a different device:

```python
# Define model stages
def create_stage1():
    stage = ml.Model()
    stage |= ml.Linear(dimension=1024)(input="input", output="hidden1")
    stage += ml.Relu()(input="hidden1", output="hidden1_act")
    return stage

def create_stage2():
    stage = ml.Model()
    stage |= ml.Linear(dimension=1024)(input="input", output="hidden2")
    stage += ml.Relu()(input="hidden2", output="hidden2_act")
    return stage

def create_stage3():
    stage = ml.Model()
    stage |= ml.Linear(dimension=10)(input="input", output="output")
    return stage

# Create pipeline model
model = ml.Model()
model |= ml.Pipeline(
    stages=[create_stage1(), create_stage2(), create_stage3()],
    device_assignment=[0, 1, 2],  # Assign stages to devices
    microbatch_size=8              # Divide batch into microbatches
)(input="input", output="output")
```

Pipeline parallelism processes different micro-batches on different stages simultaneously, achieving higher device utilization.

### Expert Parallelism (Mixture of Experts)

For models with a mixture of experts architecture:

```python
# Create a Mixture of Experts layer with experts distributed across devices
model |= ml.MoE(
    num_experts=8,
    expert_dim=1024,
    expert_distribution="one_per_device",  # Each expert on a different device
    device_mesh=(8,)                       # 8 devices for 8 experts
)(input="input", output="hidden")
```

## Optimizing Communication

Efficient communication is critical for model parallelism. Mithril provides several options to optimize it:

### Custom Communication Patterns

```python
# Specify custom collective operations for specific transfers
jax_backend = JaxBackend(
    device_mesh=(4,),
    collective_ops={
        "all_reduce": "ring",   # Use ring all-reduce
        "all_gather": "tree",   # Use tree-based all-gather
        "reduce_scatter": "hierarchical"  # Use hierarchical reduce-scatter
    }
)
```

### Communication Optimization

```python
# Enable communication optimization
jax_backend = JaxBackend(
    device_mesh=(4,),
    communication_optimization=True,  # Optimize communication patterns
    overlap_communication=True,       # Overlap computation and communication
    pipeline_communication=True       # Pipeline communication operations
)
```

## Backend-Specific Model Parallelism

### JAX Backend

JAX offers powerful tools for model parallelism, particularly with its SPMD programming model:

```python
jax_backend = JaxBackend(
    device_mesh=(2, 2),        # 2x2 device mesh
    mesh_context="spmd",       # Use SPMD programming model
    auto_sharding=True,        # Enable automatic sharding analysis
    spmd_mesh_shape=(2, 2),    # Physical mesh shape
    spmd_partition_dims=2      # Number of partition dimensions
)
```

### PyTorch Backend

PyTorch supports model parallelism through various mechanisms:

```python
torch_backend = TorchBackend(
    device_mesh=(4,),
    partition_policy="parameters",  # Partition by parameters
    use_module_hooks=True,         # Use hooks for cross-device communication
    trace_module_partitioning=True  # Enable tracing for partition debugging
)
```

### Specialized Hardware

For specialized hardware like TPUs:

```python
tpu_backend = JaxBackend(
    device_mesh=(2, 4),        # 2x4 TPU slice
    backend="tpu",             # Use TPU backend
    topology="2x4x8",          # Specify TPU topology
    memory_optimization=True   # Enable TPU memory optimization
)
```

## Memory Optimization

Model parallelism is often used to address memory limitations:

### Activation Checkpointing

```python
# Enable activation checkpointing to save memory
compiled_model = ml.compile(
    model,
    jax_backend,
    checkpointing="all"  # Checkpoint all activations
)
```

### Selective Activation Offloading

```python
# Offload activations to CPU to save GPU memory
compiled_model = ml.compile(
    model,
    torch_backend,
    activation_offload=True,  # Enable activation offloading
    offload_threshold=1e6     # Offload activations larger than 1M elements
)
```

### Mixed Precision

```python
# Use different precision for different parts of the model
jax_backend = JaxBackend(
    device_mesh=(4,),
    weight_dtype="bfloat16",    # Store weights in bfloat16
    activation_dtype="float32",  # Keep activations in float32
    compute_dtype="float32"      # Perform computation in float32
)
```

## Automatic Model Partitioning

Mithril can automatically partition your model across devices:

```python
# Enable automatic model partitioning
compiled_model = ml.compile(
    model,
    jax_backend,
    auto_partition=True,                 # Enable auto-partitioning
    partition_objective="memory_usage",  # Optimize for memory usage
    max_memory_per_device="16GB"         # Target memory per device
)
```

## Monitoring and Debugging

### Profiling Device Usage

```python
# Enable profiling
with ml.profile(backend) as prof:
    outputs = compiled_model.evaluate(params, inputs)

# Print detailed device utilization
print(prof.device_summary())

# Get communication volume
comm_stats = prof.communication_stats()
print(f"Total communication volume: {comm_stats['total_bytes']} bytes")
```

### Visualizing Model Partitioning

```python
# Visualize the model partitioning
partition_map = compiled_model.get_partition_map()
ml.visualize.plot_partition_map(partition_map, device_mesh=(4,))
```

### Memory Tracking

```python
# Track memory usage across devices
memory_usage = compiled_model.get_memory_usage(params, inputs)
for device_id, usage in enumerate(memory_usage):
    print(f"Device {device_id}: {usage['total'] / 1e9:.2f} GB")
```

## Best Practices

1. **Identify communication bottlenecks**: Profile to find excessive cross-device transfers
2. **Balance computation across devices**: Aim for similar computational load per device
3. **Minimize cross-device dependencies**: Group related operations on the same device
4. **Use appropriate sharding strategies**: Choose sharding dimensions based on operation patterns
5. **Consider memory usage**: Plan for activations, weights, and optimizer states
6. **Test with small models first**: Verify your partitioning strategy before scaling up
7. **Monitor device utilization**: Ensure all devices are well utilized

## Limitations and Challenges

1. **Increased complexity**: Model parallelism is more complex to implement than data parallelism
2. **Communication overhead**: Cross-device communication can become a bottleneck
3. **Load balancing**: Ensuring even distribution of computation is challenging
4. **Limited scaling efficiency**: Communication costs may limit how far you can scale
5. **Backend differences**: Model parallelism implementations vary between backends

## Comparison with Other Parallelism Techniques

| Feature | Model Parallelism | Data Parallelism | Pipeline Parallelism |
|---------|-------------------|------------------|----------------------|
| Memory Efficiency | High | Low | Medium |
| Communication Pattern | Activations & gradients | Gradients only | Activations & gradients |
| Implementation Complexity | High | Low | Medium |
| Scaling Efficiency | Limited by dependencies | Good | Good for deep models |
| Best For | Very large models | Smaller models, large data | Deep models |

## Examples

### Large Transformer Model with Model Parallelism

```python
import mithril as ml
from mithril.backends import TorchBackend
import numpy as np

# Create a large transformer model with model parallelism
def create_transformer_block(layer_index, device_id):
    block = ml.Model()
    
    # Self-attention on device_id
    block |= ml.MultiHeadAttention(
        num_heads=32,
        head_dim=64,
        device_id=device_id
    )(
        input=f"layer{layer_index}_input", 
        output=f"layer{layer_index}_attn"
    )
    
    # Add & Norm on device_id
    block += ml.Add()(
        inputs=[f"layer{layer_index}_input", f"layer{layer_index}_attn"], 
        output=f"layer{layer_index}_attn_res"
    )
    block += ml.LayerNorm()(
        input=f"layer{layer_index}_attn_res", 
        output=f"layer{layer_index}_norm1"
    )
    
    # FFN on next device_id
    next_device = (device_id + 1) % 4
    block += ml.FeedForward(
        hidden_dim=16384,
        device_id=next_device
    )(
        input=f"layer{layer_index}_norm1", 
        output=f"layer{layer_index}_ffn"
    )
    
    # Add & Norm on next device_id
    block += ml.Add()(
        inputs=[f"layer{layer_index}_norm1", f"layer{layer_index}_ffn"], 
        output=f"layer{layer_index}_ffn_res"
    )
    block += ml.LayerNorm()(
        input=f"layer{layer_index}_ffn_res", 
        output=f"layer{layer_index}_output"
    )
    
    return block

# Create a model with 8 transformer blocks across 4 devices
model = ml.Model()
model |= ml.Embedding(
    vocab_size=50000, 
    embedding_dim=2048,
    device_id=0
)(input="input", output="embedding")

# Add position embeddings
model += ml.PositionalEncoding(
    max_len=1024,
    device_id=0
)(input="embedding", output="pos_embedding")

# Add transformer blocks
for i in range(8):
    device_id = (i // 2) % 4  # Distribute blocks across 4 devices
    model += create_transformer_block(i, device_id)
    
    # Connect blocks
    if i == 0:
        model.connect(
            source="pos_embedding", 
            destination=f"layer{i}_input"
        )
    else:
        model.connect(
            source=f"layer{i-1}_output", 
            destination=f"layer{i}_input"
        )

# Final projection on last device
model += ml.Linear(
    dimension=50000,
    device_id=3
)(input="layer7_output", output="logits")

# Create a backend with 4 GPUs
backend = TorchBackend(
    device_mesh=(4,),
    compute_dtype="bfloat16",  # Use bfloat16 for computation
    allow_tf32=True            # Allow TF32 for matrix multiplications
)

# Compile model
compiled_model = ml.compile(
    model, 
    backend,
    checkpointing="selective",  # Use selective activation checkpointing
    partition_hooks=True        # Add hooks for profiling partition boundaries
)

# Create inputs (not sharded)
batch_size = 32
seq_length = 512
inputs = {
    "input": backend.randint(0, 50000, (batch_size, seq_length))
}

# Initialize parameters
params = compiled_model.get_parameters()

# Forward pass
outputs = compiled_model.evaluate(params, inputs)
```

## Conclusion

Model parallelism is a powerful technique for enabling the training and inference of large models that wouldn't otherwise fit on a single device. While more complex than data parallelism, it offers unique advantages for memory-constrained scenarios and can be combined with other parallelism techniques for maximum efficiency.

Mithril's flexible model composition system, combined with its device mesh and sharding capabilities, makes it possible to implement sophisticated model parallelism strategies tailored to your specific hardware configuration and model architecture. By following the best practices outlined in this guide, you can effectively distribute large models across multiple devices, enabling the training of more powerful models than would otherwise be possible.