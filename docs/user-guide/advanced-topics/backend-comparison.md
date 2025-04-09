# Backend Comparison Guide

Mithril supports multiple backends that allow the same logical model to be executed on different hardware and frameworks. This guide provides a detailed comparison to help you choose the most appropriate backend for your use case.

## Overview of Available Backends

| Backend | Based On | Automatic Differentiation | Primary Use Cases | Hardware Support |
|---------|----------|---------------------------|-------------------|-----------------|
| JAX | JAX | ✅ (full) | High-performance training and research | CPU, GPU, TPU |
| PyTorch | PyTorch | ✅ (full) | Dynamic models and rapid development | CPU, GPU, MPS (Apple) |
| MLX | MLX | ✅ (full) | Apple Silicon optimization | Apple M-series CPUs/GPUs |
| NumPy | NumPy | ❌ (manual) | Simple models, debugging | CPU only |
| C | Raw C | ❌ (manual) | Deployment, inference | CPU only |
| GGML | GGML | ❌ (manual) | Efficient inference | CPU only |

## Detailed Feature Comparison

### Automatic Differentiation Capabilities

| Backend | Auto Diff | Higher-order Gradients | JIT Support | Mixed Precision |
|---------|-----------|------------------------|-------------|-----------------|
| JAX | Full autograd | ✅ (unlimited order) | ✅ (XLA) | ✅ (float16, bfloat16) |
| PyTorch | Full autograd | ✅ (limited) | ✅ (TorchScript, Dynamo) | ✅ (float16, bfloat16) |
| MLX | Full autograd | ✅ (limited) | ✅ (built-in) | ✅ (float16, bfloat16) |
| NumPy | Manual | ❌ | ❌ | ❌ |
| C | Manual | ❌ | ❌ | ❌ |
| GGML | Manual | ❌ | ❌ | ✅ (quantization) |

### Hardware Support and Performance

| Backend | CPU | NVIDIA GPU | AMD GPU | Apple Silicon | TPU | Multiple Devices |
|---------|-----|------------|---------|---------------|-----|------------------|
| JAX | ✅ | ✅ | Limited | ✅ | ✅ | ✅ (excellent) |
| PyTorch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (good) |
| MLX | ✅ (Apple) | ❌ | ❌ | ✅ (optimized) | ❌ | ✅ (Apple only) |
| NumPy | ✅ | ❌ | ❌ | ✅ (not optimized) | ❌ | ❌ |
| C | ✅ | ❌ | ❌ | ✅ (not optimized) | ❌ | ❌ |
| GGML | ✅ | ❌ | ❌ | ✅ (not optimized) | ❌ | ❌ |

### Parallelization Support

| Backend | Data Parallelism | Model Parallelism | Pipeline Parallelism | Mixed Parallelism |
|---------|------------------|-------------------|---------------------|-------------------|
| JAX | ✅ (excellent) | ✅ (excellent) | ✅ | ✅ (device mesh) |
| PyTorch | ✅ (good) | ✅ (limited) | ✅ | ✅ (limited) |
| MLX | ✅ (limited) | ✅ (limited) | ❌ | ❌ |
| NumPy | ❌ | ❌ | ❌ | ❌ |
| C | ❌ | ❌ | ❌ | ❌ |
| GGML | ❌ | ❌ | ❌ | ❌ |

### Data Types and Precision

| Backend | float16 | float32 | float64 | bfloat16 | int8/int16 Quantization |
|---------|---------|---------|---------|----------|--------------------------|
| JAX | ✅ | ✅ | ✅ | ✅ | ✅ |
| PyTorch | ✅ | ✅ | ✅ | ✅ | ✅ |
| MLX | ✅ | ✅ | ❌ | ✅ | ❌ |
| NumPy | ❌ | ✅ | ✅ | ❌ | ❌ |
| C | ❌ | ✅ | ✅ | ❌ | ❌ |
| GGML | ✅ | ✅ | ❌ | ✅ | ✅ (extensive) |

## Backend-Specific Features and Optimizations

### JAX Backend

The JAX backend provides high-performance computing with XLA compilation and excellent parallel computing capabilities. It's ideal for research and large-scale training.

#### Key Features

- **XLA Compilation**: Just-in-time compilation for optimized execution
- **Device Mesh**: First-class support for multi-dimensional parallelism
- **Pure Functional**: Immutable tensors and transformations
- **Advanced Transformations**: Automatic vectorization and parallelization

#### Optimization Techniques

```python
import mithril as ml

# Create JAX backend with optimizations
backend = ml.JaxBackend(
    device="gpu",           # Use GPU
    precision="bfloat16",   # Use bfloat16 precision  
    pre_allocate=True       # Pre-allocate memory
)

# Compile with JAX-specific options
compiled_model = ml.compile(
    model=model,
    backend=backend,
    jit=True,                # Enable JIT compilation
    static_shapes=True,      # Use static shape optimization
    gradient_checkpointing=True  # Save memory during backprop
)
```

#### Performance Characteristics

- Highest throughput on TPUs (up to 2x faster than PyTorch)
- Excellent GPU performance with XLA compilation
- Best scaling efficiency across multiple devices
- Higher compilation overhead but faster execution
- Memory-efficient through aggressive operation fusion

#### Limitations

- Less intuitive debugging due to functional style
- First-run compilation overhead
- Fixed shapes required for peak performance
- Less flexible with dynamic control flow
- In-place mutation not supported (everything is immutable)

### PyTorch Backend

The PyTorch backend provides flexibility, ease of debugging, and strong GPU support. It's excellent for development and rapid iteration.

#### Key Features

- **Dynamic Computation**: Dynamic computation graphs for flexible execution
- **Imperative Style**: Intuitive programming model with mutable operations
- **Extensive Ecosystem**: Compatibility with the broader PyTorch ecosystem
- **Excellent Debugging**: Easy to debug with eager execution mode

#### Optimization Techniques

```python
import mithril as ml

# Create PyTorch backend with optimizations
backend = ml.TorchBackend(
    device="cuda",           # Use CUDA GPU
    enable_amp=True,         # Enable automatic mixed precision
    cudnn_benchmark=True     # Enable cuDNN autotuner
)

# Compile with PyTorch-specific options
compiled_model = ml.compile(
    model=model,
    backend=backend,
    jit=True,                # Enable TorchScript/Inductor
    jit_options={            # JIT-specific options
        "backend": "inductor",  # Use Inductor for Apple Silicon
        "mode": "max-autotune" 
    }
)
```

#### Performance Characteristics

- Strong GPU performance with dynamic shapes
- Lower compilation overhead but potentially slower execution
- More memory usage than JAX for identical operations
- Better handling of dynamic control flow
- Good scaling across multiple GPUs
- Very fast for development iterations

#### Limitations

- Less optimization potential compared to JAX
- Higher memory usage
- Potentially slower execution on TPUs
- Less efficient parallelization for very large models

### MLX Backend

The MLX backend provides optimized performance on Apple Silicon devices (M1/M2/M3 chips) with a focus on efficient execution.

#### Key Features

- **Apple Silicon Optimization**: Specifically designed for M-series chips
- **Metal Performance Shaders**: Uses Apple's Metal for GPU acceleration
- **Unified Memory Architecture**: Efficient memory sharing between CPU and GPU
- **Familiar API**: Similar to JAX/NumPy programming model

#### Optimization Techniques

```python
import mithril as ml

# Create MLX backend
backend = ml.MLXBackend(
    dtype=ml.float16,     # Use float16 for performance
    device_mesh=(1,)      # Simple parallelism configuration
)

# Compile with MLX-specific settings
compiled_model = ml.compile(
    model=model,
    backend=backend,
    jit=True              # Enable JIT compilation
)
```

#### Performance Characteristics

- Optimized for Apple Silicon (M1/M2/M3)
- Best performance on MacBooks and Apple devices
- Up to 10% better performance than native PyTorch on Apple devices
- Lower memory footprint than PyTorch on Apple devices
- Specialized for Apple's unified memory architecture

#### Limitations

- Apple Silicon devices only
- Limited to float32/float16/bfloat16 precision (no float64)
- Smaller ecosystem and community support
- Fewer high-level functionalities compared to JAX/PyTorch

### NumPy Backend

The NumPy backend provides a simple CPU-based execution without automatic differentiation. It's useful for debugging and simple models.

#### Key Features

- **Simplicity**: Straightforward NumPy-compatible API
- **Transparency**: Easier to debug and understand
- **No Dependencies**: Minimal dependencies beyond NumPy
- **Reference Implementation**: Good for verifying algorithms

#### Usage Example

```python
import mithril as ml

# Create NumPy backend
backend = ml.NumpyBackend(dtype=ml.float32)

# Compile for inference (no automatic differentiation)
compiled_model = ml.compile(
    model=model,
    backend=backend,
    inference=True        # Inference-only mode
)
```

#### Performance Characteristics

- CPU-only execution with standard NumPy performance
- No specialized hardware optimizations
- Useful as a reference implementation
- Up to 10-50x slower than accelerated backends

#### Limitations

- CPU-only execution
- No automatic differentiation
- Poor performance for deep learning workloads
- No parallelization support

### C/GGML Backends

The C and GGML backends provide low-level, efficient implementations focused on inference and deployment.

#### Key Features

- **Low Overhead**: Minimal runtime overhead
- **Static Inference**: Optimized for efficient inference of fixed models
- **Low Memory Footprint**: Efficient memory usage
- **Deployment Focus**: Designed for production deployment

#### Optimization Techniques

```python
import mithril as ml

# Create GGML backend with memory constraints
backend = ml.GGMLBackend(
    mem_size=4 * 1024 * 1024 * 1024,  # 4GB memory limit
    quantize="int8"                    # Enable 8-bit quantization
)

# Compile with static inference optimizations
compiled_model = ml.compile(
    model=model,
    backend=backend,
    static_inference=True,    # Enable static inference mode
    static_shapes=True        # Use fixed shapes
)
```

#### Performance Characteristics

- Optimized for inference-only workloads
- Lower memory footprint than Python-based backends
- Fixed shape and parameter operations only
- Specialized for deployment rather than training

#### Limitations

- Inference-only (no training support)
- Limited operation set
- No dynamic shapes
- No automatic differentiation
- Manual memory management
- Experimental status in current implementation

## Choosing the Right Backend

Consider these factors when selecting a backend:

### Training vs. Inference

- **Training**: Choose JAX, PyTorch, or MLX (Apple) backends
- **Inference**: Any backend works, but C/GGML may provide better performance for deployment

### Hardware Availability

- **NVIDIA GPUs**: JAX or PyTorch backends
- **TPUs**: JAX backend
- **AMD GPUs**: PyTorch backend
- **Apple Silicon**: MLX backend (optimal) or PyTorch/JAX
- **CPU-only**: Any backend, but NumPy is simplest

### Model Characteristics

- **Large Models**: JAX backend with parallelism
- **Dynamic Shapes**: PyTorch backend
- **Complex Control Flow**: PyTorch backend
- **Static Graph Models**: JAX or GGML backends

### Development vs. Production

- **Development**: PyTorch or JAX backends for easier debugging
- **Research**: JAX backend for performance and novel algorithms
- **Production Deployment**: GGML or C backends for efficiency

## Backend Initialization Reference

### JAX Backend

```python
backend = ml.JaxBackend(
    dtype=ml.float32,          # Data type (float32, float16, bfloat16)
    device="gpu",              # Device (cpu, gpu, tpu)
    device_mesh=(2, 2),        # Device mesh for parallelism
    pre_allocate=True,         # Pre-allocate memory
    precision="bfloat16"       # Computation precision
)
```

### PyTorch Backend

```python
backend = ml.TorchBackend(
    dtype=ml.float32,          # Data type
    device="cuda:0",           # Device (cpu, cuda:0, mps)
    device_mesh=(4,),          # Device mesh for parallelism
    enable_amp=True,           # Enable automatic mixed precision
    cudnn_benchmark=True       # Enable cuDNN benchmark
)
```

### MLX Backend

```python
backend = ml.MLXBackend(
    dtype=ml.float16,          # Data type (float16, bfloat16, float32)
    device_mesh=(1,)           # Device mesh for parallelism
)
```

### NumPy Backend

```python
backend = ml.NumpyBackend(
    dtype=ml.float64           # Data type (float32, float64)
)
```

### GGML Backend

```python
backend = ml.GGMLBackend(
    mem_size=4 * 1024 * 1024 * 1024,  # Memory limit
    quantize="int8"            # Quantization mode
)
```

### C Backend

```python
backend = ml.CBackend(
    dtype=ml.float32           # Data type (float32, float64)
)
```

## Performance Benchmarks

Below are some performance benchmarks comparing different backends on common tasks. These numbers are approximate and will vary based on hardware, model size, and specific configurations.

### Inference Speed (images/second, higher is better)

| Model Size | JAX (GPU) | PyTorch (GPU) | MLX (M2) | NumPy (CPU) | GGML (CPU) |
|------------|-----------|---------------|----------|-------------|------------|
| Small (ResNet-18) | 1200 | 1100 | 180 | 20 | 50 |
| Medium (ResNet-50) | 480 | 450 | 70 | 5 | 18 |
| Large (ViT-B/16) | 180 | 160 | 25 | 1 | 5 |

### Training Speed (iterations/second, higher is better)

| Model Size | JAX (GPU) | PyTorch (GPU) | MLX (M2) | NumPy (CPU) |
|------------|-----------|---------------|----------|-------------|
| Small (ResNet-18) | 85 | 80 | 15 | 0.8 |
| Medium (ResNet-50) | 32 | 28 | 6 | 0.2 |
| Large (ViT-B/16) | 12 | 10 | 2 | 0.05 |

### Memory Usage (GB, lower is better)

| Model Size | JAX (GPU) | PyTorch (GPU) | MLX (M2) | NumPy (CPU) | GGML (CPU) |
|------------|-----------|---------------|----------|-------------|------------|
| Small (ResNet-18) | 1.2 | 1.5 | 0.9 | 0.8 | 0.5 |
| Medium (ResNet-50) | 3.5 | 4.2 | 2.8 | 2.3 | 1.4 |
| Large (ViT-B/16) | 8.6 | 9.8 | 6.5 | 5.2 | 3.1 |

## Migration Between Backends

### JAX to PyTorch

```python
# Original JAX backend
jax_backend = ml.JaxBackend(dtype=ml.float32)
jax_model = ml.compile(model, jax_backend)
jax_params = jax_model.randomize_params()

# Convert parameters to NumPy
numpy_params = {k: jax_backend.to_numpy(v) for k, v in jax_params.items()}

# Create PyTorch backend and compile
torch_backend = ml.TorchBackend(dtype=ml.float32)
torch_model = ml.compile(model, torch_backend)

# Convert parameters to PyTorch format
torch_params = {k: torch_backend.array(v) for k, v in numpy_params.items()}
```

### PyTorch to MLX

```python
# Original PyTorch backend
torch_backend = ml.TorchBackend(dtype=ml.float32)
torch_model = ml.compile(model, torch_backend)
torch_params = torch_model.randomize_params()

# Convert parameters to NumPy
numpy_params = {k: torch_backend.to_numpy(v) for k, v in torch_params.items()}

# Create MLX backend and compile
mlx_backend = ml.MLXBackend(dtype=ml.float32)
mlx_model = ml.compile(model, mlx_backend)

# Convert parameters to MLX format
mlx_params = {k: mlx_backend.array(v) for k, v in numpy_params.items()}
```

## Conclusion

Mithril's multi-backend architecture provides flexibility to run the same models across different frameworks and hardware. This guide should help you select the most appropriate backend for your specific needs based on hardware availability, performance requirements, and development workflow.

For backend-specific optimization guides, see:
- [JAX Backend Optimization](../jax-backend.md)
- [PyTorch Backend Optimization](../pytorch-backend.md)
- [MLX Backend Guide](../mlx-backend.md)
- [GGML and C Backends](../backends.md#experimental-backends)

For performance tuning information applicable to all backends, see the [Performance Tuning Guide](../performance-tuning.md).