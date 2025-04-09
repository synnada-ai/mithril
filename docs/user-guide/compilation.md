# Compilation: The Bridge Between Design and Execution

In Mithril, compilation is the transformative process that bridges the gap between logical model design and optimized physical execution. This is where the magic happens—where your framework-agnostic model definitions are transformed into highly optimized implementations for specific hardware targets.

## The Compilation Revolution

Traditional machine learning frameworks blur the line between model definition and execution, forcing developers to embed implementation details directly in their model code. This tight coupling creates friction when moving between research, scaling, and production environments.

Mithril's compilation-based approach separates these concerns:

```python
# Define the model architecture once
model = Model()
model |= Conv2d(3, 64, kernel_size=3)
model += BatchNorm2d(64)
model += Relu()
model += Conv2d(64, 128, kernel_size=3)
model += GlobalAvgPool2d()
model += Linear(128, 10)

# Compile for different environments without changing the model
dev_model = ml.compile(model, ml.TorchBackend())       # Research
scale_model = ml.compile(model, ml.JaxBackend())       # Training at scale
deploy_model = ml.compile(model, ml.GGMLBackend())     # Deployment
```

## Basic Compilation

At its simplest, compilation takes a logical model and a backend:

```python
compiled_model = ml.compile(
    model=model,
    backend=ml.TorchBackend()
)
```

This creates a physical model—an executable implementation optimized for the specific backend.

## The Compilation Pipeline

Behind the scenes, compilation involves several sophisticated steps:

1. **Model Flattening**: Converting the hierarchical logical model into a flat graph
2. **Terminal Resolution**: Mapping named connections to physical data paths
3. **Shape Inference**: Determining tensor shapes throughout the graph
4. **Type Inference**: Resolving data types for all operations
5. **Graph Optimization**: Eliminating redundancy and dead code
6. **Code Generation**: Creating efficient backend-specific code
7. **Function Generation**: Building compiled functions for evaluation

This pipeline transforms high-level model specifications into highly optimized implementations.

## Providing Additional Information

While Mithril can infer many details automatically, you can provide additional information to guide compilation:

### Shape Information

```python
# Specify input shapes for better optimization
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 3, 224, 224]}  # Batch of 32 images (224x224)
)
```

### Type Information

```python
# Specify specific data types
compiled_model = ml.compile(
    model=model,
    backend=backend,
    types={"input": ml.float16}  # Use half-precision for input
)
```

### Static Shapes

```python
# Optimize for fixed input shapes
compiled_model = ml.compile(
    model=model,
    backend=backend,
    shapes={"input": [32, 3, 224, 224]},
    static_shapes=True  # Enable static shape optimization
)
```

## JIT Compilation

Just-in-time compilation can significantly improve performance:

```python
# Enable JIT compilation
compiled_model = ml.compile(
    model=model,
    backend=ml.JaxBackend(),
    jit=True  # Use XLA compilation
)
```

Each backend implements JIT differently:
- **JAX**: Uses XLA compilation for optimized execution
- **PyTorch**: Uses TorchScript or Torch Inductor for optimization
- **MLX**: Uses MLX's built-in compilation for Apple Silicon

## Backend-Specific Optimizations

Different backends offer specialized optimizations:

### JAX Optimizations

```python
# JAX-specific compilation options
jax_model = ml.compile(
    model=model,
    backend=ml.JaxBackend(),
    jit=True,
    jit_options={
        "static_argnums": (0, 1),  # Arguments that should be treated as static
        "donate_argnums": (0,),    # Arguments that can be overwritten (buffer reuse)
    }
)
```

### PyTorch Optimizations

```python
# PyTorch-specific compilation options
torch_model = ml.compile(
    model=model,
    backend=ml.TorchBackend(),
    jit=True,
    jit_options={
        "backend": "inductor",  # Or "eager", "aot_eager", etc.
        "mode": "max-autotune"  # Performance optimization level
    }
)
```

## Memory Optimization

For large models, memory optimization is crucial:

```python
# Enable memory optimization
compiled_model = ml.compile(
    model=model,
    backend=backend,
    gradient_checkpointing=True,  # Trade computation for memory
    checkpoint_segments=4,        # Number of segments for checkpointing
    memory_planning="greedy"      # Memory allocation strategy
)
```

## Mixed Precision

Reduce memory usage and speed up computation with mixed precision:

```python
# Enable mixed precision
compiled_model = ml.compile(
    model=model,
    backend=ml.JaxBackend(precision="bfloat16"),
    mixed_precision=True  # Use different precision for different operations
)
```

## Static Inference

For deployment scenarios, static inference provides maximum optimization:

```python
# Optimize for inference
compiled_model = ml.compile(
    model=model,
    backend=ml.GGMLBackend(),
    inference=True,       # Inference-only mode
    static_inference=True  # Use static optimizations
)
```

## Output Code Generation

You can also generate code for inspection or separate compilation:

```python
# Generate code to a file
compiled_model = ml.compile(
    model=model,
    backend=backend,
    file_path="generated_model.py"  # Path to write generated code
)
```

## The Compiled Model API

Once compiled, a physical model provides a consistent API across all backends:

```python
# Initialize parameters
params = compiled_model.randomize_params()

# Run forward pass
outputs = compiled_model.evaluate(params, inputs)

# Compute gradients
outputs, gradients = compiled_model.evaluate(
    params, 
    inputs, 
    output_gradients=output_grads
)

# Get shape information
shapes = compiled_model.get_shapes()

# Get type information
types = compiled_model.get_types()
```

## Advanced Compilation Techniques

### Graph Optimization

```python
# Enable specific optimizations
compiled_model = ml.compile(
    model=model,
    backend=backend,
    optimizations={
        "constant_folding": True,      # Pre-compute constant expressions
        "operator_fusion": True,       # Combine consecutive operations
        "dead_code_elimination": True, # Remove unused computations
        "common_subexpression": True   # Reuse results of duplicate expressions
    }
)
```

### Hardware Targeting

```python
# Target specific hardware
compiled_model = ml.compile(
    model=model,
    backend=ml.JaxBackend(device="gpu"),
    target_hardware="nvidia_a100",  # Specific GPU architecture
    optimize_for="throughput"       # Prioritize throughput over latency
)
```

## Compilation Best Practices

1. **Compile once, run many times**: Compilation can be expensive, so compile once and reuse the compiled model
2. **Start with small shapes**: Begin with small input sizes to ensure compilation works correctly
3. **Provide shape information**: Explicit shapes enable better optimization
4. **Use backend-specific options wisely**: Each backend has unique capabilities
5. **Profile before optimizing**: Measure performance before applying optimizations
6. **Balance precision and speed**: Lower precision often means faster execution
7. **Consider deployment targets early**: Compilation options affect deployment capabilities

## Troubleshooting Compilation

### Common Errors and Solutions

**Shape Inference Failures**
```
Error: Cannot infer shape for tensor 'hidden_1'
```
Solution: Provide explicit shapes for ambiguous terminals

**Type Inconsistencies**
```
Error: Type mismatch: expected float32, got float64
```
Solution: Check backend dtype settings and ensure consistent types

**Memory Errors**
```
Error: CUDA out of memory
```
Solution: Enable gradient checkpointing or reduce batch size

**Compilation Timeout**
```
Error: Compilation timed out after 300 seconds
```
Solution: Simplify model or disable complex optimizations

## Conclusion

Compilation is the heart of Mithril's "define once, run anywhere" philosophy. By separating model architecture from execution details, Mithril enables a fundamentally more flexible and efficient ML development process.

For advanced compilation techniques, see [Advanced Compilation](advanced-compilation.md).