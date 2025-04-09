# Mixed Parallelism

This guide explains how to combine different parallelism strategies in Mithril to maximize performance and efficiency.

## Overview

Mixed parallelism involves using multiple parallelism techniques simultaneously:

- **Data Parallelism**: Replicating the model across devices and processing different data batches
- **Model Parallelism**: Splitting the model across devices for memory or computational benefits
- **Pipeline Parallelism**: Dividing the model into sequential stages executed on different devices
- **Tensor Parallelism**: Partitioning individual tensors across devices for parallel computation

By combining these approaches, you can:

- Scale to larger models and batch sizes than any single technique allows
- Optimize resource utilization across heterogeneous hardware
- Balance computation, memory usage, and communication overhead
- Achieve higher throughput and lower latency

## Basic Concept

Mixed parallelism typically employs multi-dimensional device meshes to represent different parallelism dimensions:

![Mixed Parallelism Diagram](../assets/mixed_parallelism.png)

For example, in a 2D device mesh:
- One dimension might represent data parallelism
- The other dimension might represent model parallelism

## Implementing Mixed Parallelism in Mithril

### Multi-dimensional Device Mesh

First, create a multi-dimensional device mesh:

```python
import mithril as ml
from mithril.backends import JaxBackend, TorchBackend

# Create a 2D device mesh (4x2 = 8 devices)
# - First dimension (4): Data parallelism
# - Second dimension (2): Model parallelism
jax_backend = JaxBackend(device_mesh=(4, 2))
torch_backend = TorchBackend(device_mesh=(4, 2))
```

### Data + Model Parallelism

Combine data and model parallelism by sharding data across one dimension and model parts across another:

```python
# Create a model with combined data and model parallelism
model = ml.Model()

# Large embedding layer split across model dimension
model |= ml.Embedding(
    vocab_size=100000,
    embedding_dim=4096,
    weight_shard_spec=(0, None),  # Shard weights along vocab dimension
    device_mesh=(1, 2)            # Across model dimension only
)(input="input", output="embedding")

# Process embedding on all devices
model += ml.Linear(
    dimension=2048,
    weight_shard_spec=(0, 1),     # Shard weights along both dimensions
    device_mesh=(4, 2)            # Use all devices
)(input="embedding", output="hidden1")

model += ml.Relu()(input="hidden1", output="hidden1_act")

# Final output layer on data-parallel dimension only
model += ml.Linear(
    dimension=1000,
    weight_shard_spec=(0, None),  # Shard weights along input dimension
    device_mesh=(4, 1)            # Across data dimension only
)(input="hidden1_act", output="output")
```

### Creating Sharded Inputs

For mixed parallelism, shard inputs only across the data parallelism dimension:

```python
# Create inputs sharded across data parallelism dimension only
batch_size = 128  # Total batch size
inputs = {
    "input": jax_backend.randint(
        0, 100000, 
        (batch_size, 32),  # [batch_size, sequence_length]
        device_mesh=(4, 2),      # Full device mesh
        tensor_split=(0, None)   # Split along batch dimension only
    )
}
```

### Compiling and Running

Compile and run the model as usual:

```python
# Compile model
compiled_model = ml.compile(model, jax_backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Forward pass
outputs = compiled_model.evaluate(params, inputs)
```

## Advanced Mixed Parallelism Strategies

### Data + Pipeline Parallelism

Combine data parallelism with pipeline parallelism for deep models:

```python
# Define pipeline stages
def create_stage(stage_id):
    stage = ml.Model()
    stage |= ml.Linear(dimension=1024)(input=f"stage{stage_id}_input", output=f"stage{stage_id}_hidden")
    stage += ml.Relu()(input=f"stage{stage_id}_hidden", output=f"stage{stage_id}_output")
    return stage

# Create pipeline model with data parallelism
model = ml.Model()
model |= ml.Pipeline(
    stages=[create_stage(i) for i in range(4)],
    device_assignment=[0, 1, 2, 3],  # Pipeline stages on devices 0-3
    microbatch_size=8,               # Use microbatches for pipelining
    data_parallel=True,              # Enable data parallelism
    data_parallel_devices=2          # 2 devices per pipeline stage (8 total)
)(input="input", output="output")
```

### 3D Parallelism

For very large models, use 3D parallelism (data + model + pipeline):

```python
# Create a 3D device mesh (2x2x2 = 8 devices)
jax_backend = JaxBackend(device_mesh=(2, 2, 2))
# - First dimension (2): Data parallelism
# - Second dimension (2): Tensor parallelism
# - Third dimension (2): Pipeline parallelism

# Create a model using 3D parallelism
model = ml.Model()

# Pipeline stage 0 (devices [0,0,0], [0,1,0], [1,0,0], [1,1,0])
stage0 = ml.Model()
stage0 |= ml.Embedding(
    vocab_size=100000,
    embedding_dim=4096,
    weight_shard_spec=(0, None),         # Shard weights along vocab dimension
    device_mesh=(2, 2, 1),               # Use first 4 devices
    device_mesh_partition=(True, True, False)  # Use first 2 dimensions
)(input="input", output="embedding")

# Pipeline stage 1 (devices [0,0,1], [0,1,1], [1,0,1], [1,1,1])
stage1 = ml.Model()
stage1 |= ml.MultiHeadAttention(
    num_heads=32,
    head_dim=128,
    weight_shard_spec=[(0, 1), (0, 1)],   # Shard weights in both dimensions
    device_mesh=(2, 2, 1),                # Use second 4 devices
    device_mesh_partition=(True, True, False)  # Use first 2 dimensions
)(input="input", output="attention")

stage1 += ml.Linear(
    dimension=1000,
    weight_shard_spec=(0, 1),             # Shard weights in both dimensions
    device_mesh=(2, 2, 1),                # Use all devices in first 2 dimensions
    device_mesh_partition=(True, True, False)
)(input="attention", output="output")

# Combine stages with pipeline parallelism
model |= ml.Pipeline(
    stages=[stage0, stage1],
    device_assignment=[0, 1],            # Pipeline stages on third dimension
    device_mesh=(2, 2, 2),               # Full 3D mesh
    device_mesh_partition=(False, False, True)  # Use third dimension for pipeline
)(input="input", output="output")
```

### Mixture of Experts with Data Parallelism

Combine Mixture of Experts (model parallelism) with data parallelism:

```python
# Create a 2D device mesh (4x8 = 32 devices)
backend = TorchBackend(device_mesh=(4, 8))
# - First dimension (4): Data parallelism
# - Second dimension (8): Expert parallelism

# Create a model with MoE + data parallelism
model = ml.Model()
model |= ml.Linear(dimension=1024)(input="input", output="hidden")
model += ml.Relu()(input="hidden", output="hidden_act")

# Mixture of Experts layer with experts distributed across second dimension
model += ml.MoE(
    num_experts=8,
    expert_dim=2048,
    expert_distribution="one_per_device",  # Each expert on one device in dim 1
    device_mesh=(4, 8),                    # Full device mesh
    device_mesh_partition=(False, True)    # Use second dimension for experts
)(input="hidden_act", output="moe_output")

model += ml.Linear(dimension=10)(input="moe_output", output="output")
```

## Optimizing Mixed Parallelism

### Communication Optimization

In mixed parallelism, optimizing communication is critical:

```python
# Configure communication patterns based on device mesh topology
jax_backend = JaxBackend(
    device_mesh=(4, 2),
    collective_ops={
        "all_reduce": {
            "data_parallel": "ring",      # Ring all-reduce for data parallel dim
            "model_parallel": "tree"       # Tree all-reduce for model parallel dim
        },
        "all_gather": {
            "data_parallel": "hierarchical",  # Hierarchical for data parallel dim
            "model_parallel": "direct"        # Direct for model parallel dim
        }
    },
    mesh_topology="torus"  # Optimize for torus network topology
)
```

### Memory Optimization

Tailor memory optimizations for mixed parallelism:

```python
# Configure memory optimizations for mixed parallelism
compiled_model = ml.compile(
    model,
    jax_backend,
    memory_planning="greedy",        # Use greedy memory planning
    rematerialization=True,          # Enable rematerialization (recomputation)
    rematerialization_policy={
        "data_parallel": "minimal",  # Minimal rematerialization for data parallel
        "model_parallel": "full"     # Full rematerialization for model parallel
    },
    spill_to_host=True,              # Allow spilling to host memory
    spill_threshold_mb=1024          # Spill tensors larger than 1GB
)
```

### Automatic Mixed Parallelism

Let Mithril automatically determine the optimal parallelism strategy:

```python
# Enable automatic mixed parallelism
compiled_model = ml.compile(
    model,
    jax_backend,
    auto_partition=True,                 # Enable auto-partitioning
    partition_objective="throughput",    # Optimize for throughput
    partition_search_space="mixed",      # Search mixed parallelism strategies
    partition_constraints={
        "max_memory_per_device": "16GB",  # Memory constraint
        "max_all_reduce_size": "1GB"      # Communication constraint
    }
)
```

## Backend-Specific Mixed Parallelism

### JAX Backend

JAX has excellent support for mixed parallelism:

```python
# Configure JAX backend for mixed parallelism
jax_backend = JaxBackend(
    device_mesh=(4, 2),
    mesh_axis_names=["data", "model"],  # Name mesh dimensions
    pjit_mesh=True,                     # Use pjit for partitioning
    xmap_mesh=False,                    # Don't use xmap
    pjit_lowering_strategy="auto",      # Automatic lowering strategy
    auto_sharding_config={
        "force_data_parallel": False,   # Allow mixed strategies
        "allow_mixed_mesh_models": True  # Enable mixed mesh models
    }
)
```

### PyTorch Backend

PyTorch supports mixed parallelism through various mechanisms:

```python
# Configure PyTorch backend for mixed parallelism
torch_backend = TorchBackend(
    device_mesh=(4, 2),
    distributed_backend="nccl",        # Use NCCL for communication
    parallel_mode="fully_sharded",     # Use fully sharded data parallelism
    tensor_parallel_config={
        "tp_size": 2,                  # Tensor parallel size
        "tp_mode": "column",           # Column parallelism for tensors
        "overlap_comm": True           # Overlap communication with computation
    },
    fsdp_config={
        "sharding_strategy": "SHARD_GRAD_OP",  # Shard gradients and optimizer states
        "mixed_precision": True,               # Use mixed precision
        "flatten_parameters": True             # Flatten parameters for better efficiency
    }
)
```

## Performance Monitoring and Tuning

### Profiling Mixed Parallelism

```python
# Enable detailed profiling for mixed parallelism
with ml.profile(jax_backend, 
                profile_communication=True,  # Profile communication
                profile_memory=True,         # Profile memory usage
                profile_compute=True         # Profile computation
               ) as prof:
    outputs = compiled_model.evaluate(params, inputs)

# Print profiling results
print(prof.summary())

# Get detailed communication breakdown by dimension
comm_by_dim = prof.communication_by_dimension()
print(f"Data parallel communication: {comm_by_dim['data_parallel']} bytes")
print(f"Model parallel communication: {comm_by_dim['model_parallel']} bytes")
```

### Visualizing Device Utilization

```python
# Visualize device utilization across mesh dimensions
utilization_map = prof.device_utilization_map()
ml.visualize.plot_device_utilization(utilization_map, device_mesh=(4, 2))

# Plot communication patterns
comm_graph = prof.communication_graph()
ml.visualize.plot_communication_graph(comm_graph, device_mesh=(4, 2))
```

### Auto-tuning Mixed Parallelism

```python
# Automatically tune mixed parallelism strategy
best_strategy = ml.auto_tune(
    model,
    jax_backend,
    input_shapes={"input": [32, 128]},
    tuning_objective="throughput",     # Optimize for throughput
    tuning_budget_hours=2,             # Run tuning for 2 hours
    strategy_space="mixed_parallel"    # Search mixed parallelism strategies
)

# Apply the best strategy
compiled_model = ml.compile(
    model,
    jax_backend,
    partition_strategy=best_strategy
)
```

## Best Practices

1. **Match parallelism strategy to model architecture**:
   - Data parallelism: For batch-independent models with moderate size
   - Model parallelism: For very large models with memory constraints
   - Pipeline parallelism: For deep sequential models
   - Tensor parallelism: For operations with massive tensors

2. **Consider hardware topology**:
   - Place heavily communicating devices on the same switch
   - Use high-bandwidth connections for model-parallel dimension
   - Consider NVLink/InfiniBand topology when designing device meshes

3. **Balance dimensions appropriately**:
   - Data-parallel dimension: Often larger for better scaling
   - Model-parallel dimension: Keep smaller to minimize communication
   - Pipeline-parallel dimension: Balance pipeline depth vs. bubble overhead

4. **Optimize communication patterns**:
   - Minimize cross-mesh-dimension communication
   - Use efficient collective operations
   - Consider hierarchical communication for large clusters

5. **Monitor and tune regularly**:
   - Profile to identify bottlenecks
   - Adjust strategy as model architecture evolves
   - Regularly benchmark different configurations

## Example: Large Transformer with Mixed Parallelism

```python
import mithril as ml
from mithril.backends import JaxBackend
import numpy as np

# Create a 3D device mesh (4x2x2 = 16 devices)
# - Dim 0 (4): Data parallelism
# - Dim 1 (2): Tensor parallelism
# - Dim 2 (2): Pipeline parallelism
backend = JaxBackend(
    device_mesh=(4, 2, 2),
    dtype="bfloat16",              # Use bfloat16 for computation
    mesh_axis_names=["data", "tensor", "pipeline"]  # Name dimensions
)

# Create a large transformer model with mixed parallelism
def create_transformer_block(layer_idx, pipeline_stage):
    block = ml.Model()
    
    # Self-attention with tensor parallelism
    block |= ml.MultiHeadAttention(
        num_heads=32,
        head_dim=128,
        weight_shard_spec=[(0, 1), None, None],  # Shard QKV weights
        device_mesh=(4, 2, 1),                   # Use data and tensor dims
        device_mesh_partition=(True, True, False)
    )(
        input=f"layer{layer_idx}_input", 
        output=f"layer{layer_idx}_attn"
    )
    
    # Residual connection and layer norm
    block += ml.Add()(
        inputs=[f"layer{layer_idx}_input", f"layer{layer_idx}_attn"], 
        output=f"layer{layer_idx}_attn_res"
    )
    block += ml.LayerNorm()(
        input=f"layer{layer_idx}_attn_res", 
        output=f"layer{layer_idx}_norm1"
    )
    
    # Feed-forward with tensor parallelism
    block += ml.FeedForward(
        hidden_dim=16384,
        weight_shard_spec=[(0, 1), (1, 0)],  # Shard FF weights
        device_mesh=(4, 2, 1),               # Use data and tensor dims
        device_mesh_partition=(True, True, False)
    )(
        input=f"layer{layer_idx}_norm1", 
        output=f"layer{layer_idx}_ffn"
    )
    
    # Residual connection and layer norm
    block += ml.Add()(
        inputs=[f"layer{layer_idx}_norm1", f"layer{layer_idx}_ffn"], 
        output=f"layer{layer_idx}_ffn_res"
    )
    block += ml.LayerNorm()(
        input=f"layer{layer_idx}_ffn_res", 
        output=f"layer{layer_idx}_output"
    )
    
    return block

# Create pipeline stages (8 layers per stage)
stage1 = ml.Model()

# Embedding with tensor parallelism
stage1 |= ml.Embedding(
    vocab_size=50000,
    embedding_dim=4096,
    weight_shard_spec=(0, None),         # Shard along vocab dimension
    device_mesh=(4, 2, 1),               # Use data and tensor dims
    device_mesh_partition=(True, True, False)
)(input="input", output="embedding")

# Position encoding
stage1 += ml.PositionalEncoding(
    max_len=2048
)(input="embedding", output="pos_embedding")

# First 8 transformer layers
for i in range(8):
    stage1 += create_transformer_block(i, 0)
    
    # Connect blocks
    if i == 0:
        stage1.connect(
            source="pos_embedding", 
            destination=f"layer{i}_input"
        )
    else:
        stage1.connect(
            source=f"layer{i-1}_output", 
            destination=f"layer{i}_input"
        )

# Second pipeline stage with 8 more layers
stage2 = ml.Model()
for i in range(8, 16):
    stage2 += create_transformer_block(i, 1)
    
    # Connect blocks (within stage2)
    if i == 8:
        # This will be connected via pipeline
        pass
    else:
        stage2.connect(
            source=f"layer{i-1}_output", 
            destination=f"layer{i}_input"
        )

# Final prediction head
stage2 += ml.Linear(
    dimension=50000,
    weight_shard_spec=(0, 1),           # Shard weights
    device_mesh=(4, 2, 1),              # Use data and tensor dims
    device_mesh_partition=(True, True, False)
)(input="layer15_output", output="logits")

# Combine stages with pipeline parallelism
model = ml.Model()
model |= ml.Pipeline(
    stages=[stage1, stage2],
    stage_to_mesh_dim=2,               # Use dim 2 for pipeline
    microbatch_size=4,                 # Use microbatches
    device_mesh=(4, 2, 2)              # Full 3D mesh
)(input="input", output="logits")

# Compile model with mixed parallelism optimizations
compiled_model = ml.compile(
    model,
    backend,
    checkpointing="selective",           # Selective activation checkpointing
    rematerialization=True,              # Enable rematerialization
    memory_planning="greedy",            # Greedy memory planning
    communication_optimization=True,     # Optimize communication patterns
    pipeline_schedule="gpipe"            # Use GPipe pipeline schedule
)

# Create inputs sharded only across data parallel dimension
batch_size = 128
seq_length = 1024
inputs = {
    "input": backend.randint(
        0, 50000, 
        (batch_size, seq_length),
        device_mesh=(4, 2, 2),
        tensor_split=(0, None, None)  # Split only along data dimension
    )
}

# Initialize parameters
params = compiled_model.get_parameters()

# Forward pass
outputs = compiled_model.evaluate(params, inputs)
```

## Common Challenges and Solutions

### Challenge: Communication Bottlenecks

**Solution**: Minimize cross-dimension communication and optimize collective operations:

```python
# Optimize communication patterns
backend = JaxBackend(
    device_mesh=(4, 2),
    collective_optimization=True,           # Enable collective optimization
    all_reduce_fusion=True,                 # Fuse all-reduce operations
    communication_overlap_strategy="auto",  # Automatically overlap communication
    network_topology_aware=True             # Be aware of network topology
)
```

### Challenge: Load Balancing

**Solution**: Ensure even work distribution across devices:

```python
# Enable load balancing
compiled_model = ml.compile(
    model,
    backend,
    load_balancing=True,                     # Enable load balancing
    load_balancing_algorithm="greedy",       # Use greedy load balancing
    profiling_based_partitioning=True,       # Use profiling for better partitioning
    recomputation_granularity="operation"    # Fine-grained recomputation control
)
```

### Challenge: Memory Efficiency

**Solution**: Use advanced memory optimizations:

```python
# Configure memory optimizations
compiled_model = ml.compile(
    model,
    backend,
    memory_planning="dynamic",             # Dynamic memory planning
    activation_checkpointing="selective",  # Selective checkpointing
    gradient_accumulation=True,            # Enable gradient accumulation
    gradient_accumulation_steps=8,         # Accumulate over 8 steps
    mixed_precision_policy={
        "params": "bfloat16",              # Store parameters in bfloat16
        "compute": "float32",              # Compute in float32
        "gradients": "float32"             # Store gradients in float32
    }
)
```

## Conclusion

Mixed parallelism combines the strengths of different parallelism strategies to enable training larger models with better resource utilization. By carefully designing multi-dimensional device meshes and parallelism strategies, you can scale to very large models and datasets while maintaining high performance.

Mithril's flexible parallelism system makes it possible to implement sophisticated mixed parallelism strategies tailored to your specific hardware configuration and model architecture. The ability to specify sharding across different mesh dimensions, combined with automatic optimization capabilities, provides a powerful framework for scaling machine learning models to unprecedented sizes.